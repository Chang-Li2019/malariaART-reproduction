import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import OrderedDict
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esm3_model_tokenizers  # esm3 sequence tokenizer is the same as esmc tokenizer
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer as EsmFunctionTokenizer
from esm.pretrained import ESM3_function_decoder_v0, ESM3_structure_decoder_v0
from esm3_client import get_esm3_client
from esm3_helpers import euclidean_distance, manhattan_distance, cosine_similarity
from esm3_intervals import get_intervals_and_weights

import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ESM-3 Open Small is a gated model on HuggingFace. Authenticate once before
# first use with `huggingface-cli login`, or set the HF_TOKEN environment
# variable; from_pretrained picks the token up automatically. Do not hardcode a
# token here.

from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

class ESM3Utils:
    def __init__(self, config_dict=dict(), max_len=1500, esmc_client=None, device=None):
        """
        Initialize ESM3Utils with GPU support
        
        Args:
            config_dict: Configuration dictionary for logits
            max_len: Maximum sequence length
            esmc_client: Optional ESMC client
            device: Device to use (defaults to CUDA)
        """
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.client = get_esm3_client(device)
        self.tokenizers = get_esm3_model_tokenizers()
        self.ordered_tokens = OrderedDict(sorted(self.tokenizers.sequence.vocab.items(), key=lambda x: x[1]))
        self.aa_tokenizers = {k: v for k, v in self.tokenizers.sequence.vocab.items() 
                              if k not in self.tokenizers.sequence.all_special_tokens + ['.', '-', 'X', 'B', 'Z', 'U', 'O']}
        self.aa_tokenizers = OrderedDict(sorted(self.aa_tokenizers.items(), key=lambda x: x[1]))

        self.logits_config = LogitsConfig(
            sequence=config_dict.get('sequence', False),
            structure=config_dict.get('structure', False),
            secondary_structure=config_dict.get('secondary_structure', False),
            sasa=config_dict.get('sasa', False),
            function=config_dict.get('function', False),
            residue_annotations=config_dict.get('residue_annotations', False),
            return_embeddings=config_dict.get('return_embeddings', False)
        )
        self.max_len = max_len
        self.esmc_client = esmc_client
        self.ref_embeddings = None
        self.ref_centroid = None
        self.knn_model = None
        self.gmm_model = None
        self.attention_weights = None

    def get_protein_tensor(self, sequence):
        """Get protein tensor on GPU"""
        try:
            protein = ESMProtein(sequence=sequence)
            protein_tensor = self.client.encode(protein)
            return protein_tensor
        except Exception as e:
            print(f"Error in get_protein_tensor: {str(e)}")
            return None

    @torch.no_grad()
    def get_logits(self, sequence):
        """Get logits on GPU"""
        try:
            protein_tensor = self.get_protein_tensor(sequence)
            output = self.client.logits(protein_tensor, self.logits_config)
            return output
        except Exception as e:
            print(f"Error in get_logits: {str(e)}")
            return None

    @torch.no_grad()
    def get_individual_outputs(self, sequence):
        """
        Get individual outputs from the ESM-3 model.
        
        Args:
            sequence: Input protein sequence
            max_chunk_size: Maximum sequence length to process at once
        """
        try:
            # Clear CUDA cache before processing
            torch.cuda.empty_cache()
            
            # Process with bfloat16 on GPU
            dtype = torch.bfloat16
            
            # Convert model parameters
            def convert_model_dtype(model):
                for param in model.parameters():
                    param.data = param.data.cuda().to(dtype=dtype)
                    if hasattr(param, '_grad') and param._grad is not None:
                        param._grad.data = param._grad.data.cuda().to(dtype=dtype)
            
            if hasattr(self.client, 'encoder'):
                convert_model_dtype(self.client.encoder)
            if hasattr(self.client, 'decoder'):
                convert_model_dtype(self.client.decoder)
            
            # Create protein tensor and set config
            protein = ESMProtein(sequence=sequence)
            self.logits_config.return_embeddings = True
            self.logits_config.sequence = True
            self.logits_config.structure = True
            self.logits_config.secondary_structure = True
            self.logits_config.sasa = True
            self.logits_config.function = True
            self.logits_config.residue_annotations = True

            print(f"Encoding protein sequence of length {len(sequence)}...")
            protein_tensor = self.client.encode(protein)
            
            # Process with memory management
            try:
                outputs = self.client.logits(protein_tensor, self.logits_config)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("GPU OOM detected, clearing cache and retrying...")
                    torch.cuda.empty_cache()
                    outputs = self.client.logits(protein_tensor, self.logits_config)
                else:
                    raise e
                    
            print("Successfully processed outputs")
            return outputs
            
        except Exception as e:
            print(f"Error in get_individual_outputs: {str(e)}")
            print(f"Sequence length: {len(sequence)}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            
            # Try to free memory in case of GPU OOM
            torch.cuda.empty_cache()
            return None

    def crop_variant(self, ref_seq, alt_seq, ref_start):
        left_pos = ref_start - 1
        offset = len(ref_seq) - len(alt_seq)
        start_pos = int(left_pos - self.max_len / 2)
        end_pos1 = int(left_pos + self.max_len / 2) - min(start_pos, 0) + min(offset, 0)
        end_pos2 = int(left_pos + self.max_len / 2) - min(start_pos, 0) - max(offset, 0)
        if start_pos < 0: start_pos = 0
        if end_pos1 > len(ref_seq): end_pos1 = len(ref_seq)
        if end_pos2 > len(alt_seq): end_pos2 = len(alt_seq)
        if start_pos > 0 and max(end_pos2, end_pos1) - start_pos < self.max_len:
            start_pos = max(0, max(end_pos2, end_pos1) - self.max_len)

        return ref_seq[start_pos:end_pos1], alt_seq[start_pos:end_pos2], start_pos - ref_start

    def get_ESMC_LL(self, sequence, return_embeddings=False):
        protein_tensor = self.get_protein_tensor(sequence)
        output = self.esmc_client.logits(protein_tensor, self.logits_config)
        s = output.logits.sequence.squeeze()[1:-1, :33].cpu()
        s = s.to(torch.float32)
        WTlogits = pd.DataFrame(s, columns=self.ordered_tokens, index=list(sequence)).T.iloc[4:29]
        WTlogits.columns = [j.split('.')[0] + ' ' + str(i + 1) for i, j in enumerate(WTlogits.columns)]
        wt_norm = np.diag(WTlogits.loc[[i.split(' ')[0] for i in WTlogits.columns]])
        LLR = WTlogits - wt_norm
        if return_embeddings:
            return LLR, wt_norm.sum(), output.embeddings.squeeze()[1:-1, :].cpu().numpy()
        else:
            return LLR
    
    def get_ESMC6b_LL(self, sequence, embed_reduce_fn=np.mean):
        if len(sequence) > 2046:
            ints, M, M_norm = get_intervals_and_weights(len(sequence), min_overlap=512, max_len=2046, s=20)
            dt = [''.join(list(np.array(list(sequence))[idx])) for idx in ints]

            embedding_parts = []
            logit_parts = []
            for dt_ in dt:
                with torch.no_grad():
                    protein_tensor = self.esmc_client.encode(ESMProtein(sequence=dt_))
                    output = self.esmc_client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
                    s = output.logits.sequence.squeeze()[:, :33].cpu()
                    results_ = torch.log_softmax(input=s, dim=-1)
                    logit_parts.append(results_.numpy()[1:-1, :])
                    e = output.embeddings.squeeze()[1:-1, :].cpu().numpy()
                    embedding_parts.append(e)

            embeddings_full = np.zeros((len(sequence), e.shape[1]))
            logits_full = np.zeros((len(sequence), 33))

            for i in range(len(ints)):
                logit = np.zeros((len(sequence), 33))
                logit[ints[i]] = logit_parts[i].copy()
                logit = np.multiply(logit.T, M_norm[i, :]).T
                logits_full += logit

                embedding = np.zeros((len(sequence), e.shape[1]))
                embedding[ints[i]] = embedding_parts[i].copy()
                embedding = np.multiply(embedding.T, M_norm[i, :]).T
                embeddings_full += embedding

            WTlogits = pd.DataFrame(logits_full, columns=self.ordered_tokens, index=list(sequence)).T.iloc[4:29]
            WTlogits.columns = [j.split('.')[0] + ' ' + str(i + 1) for i, j in enumerate(WTlogits.columns)]
            wt_norm = np.diag(WTlogits.loc[[i.split(' ')[0] for i in WTlogits.columns]])
            LLR = WTlogits - wt_norm
            return LLR, embed_reduce_fn(embeddings_full, axis=0)

        else:
            protein_tensor = self.esmc_client.encode(ESMProtein(sequence=sequence))
            output = self.esmc_client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            s = output.logits.sequence.squeeze()[1:-1, :33].cpu()
            WTlogits = pd.DataFrame(s, columns=self.ordered_tokens, index=list(sequence)).T.iloc[4:29]
            WTlogits.columns = [j.split('.')[0] + ' ' + str(i + 1) for i, j in enumerate(WTlogits.columns)]
            wt_norm = np.diag(WTlogits.loc[[i.split(' ')[0] for i in WTlogits.columns]])
            LLR = WTlogits - wt_norm
            return LLR, embed_reduce_fn(output.embeddings.squeeze()[1:-1, :].cpu().numpy(), axis=0)


    def get_LL(self, sequence, embed_reduce_fn=np.mean, axis=0, return_embeddings=False):
        if len(sequence) > self.max_len:
            ints, M, M_norm = get_intervals_and_weights(len(sequence), min_overlap=self.max_len//2, max_len=self.max_len, s=20)
            dt = [''.join(list(np.array(list(sequence))[idx])) for idx in ints]

            embedding_parts = []
            logit_parts = []
            for dt_ in dt:
                with torch.no_grad():
                    output = self.get_logits(dt_)
                    s = output.logits.sequence.squeeze()[:, :33].cpu()
                    results_ = torch.log_softmax(input=s, dim=-1)
                    logit_parts.append(results_.numpy()[1:-1, :])
                    e = output.embeddings.squeeze()[1:-1, :].cpu().numpy()
                    embedding_parts.append(e)

            embeddings_full = np.zeros((len(sequence), e.shape[1]))
            logits_full = np.zeros((len(sequence), 33))

            for i in range(len(ints)):
                logit = np.zeros((len(sequence), 33))
                logit[ints[i]] = logit_parts[i].copy()
                logit = np.multiply(logit.T, M_norm[i, :]).T
                logits_full += logit

                embedding = np.zeros((len(sequence), e.shape[1]))
                embedding[ints[i]] = embedding_parts[i].copy()
                embedding = np.multiply(embedding.T, M_norm[i, :]).T
                embeddings_full += embedding

            WTlogits = pd.DataFrame(logits_full, columns=self.ordered_tokens, index=list(sequence)).T.iloc[4:29]
            WTlogits.columns = [j.split('.')[0] + ' ' + str(i + 1) for i, j in enumerate(WTlogits.columns)]
            wt_norm = np.diag(WTlogits.loc[[i.split(' ')[0] for i in WTlogits.columns]])
            LLR = WTlogits - wt_norm
            if return_embeddings:
                return embeddings_full
            else:
                return LLR, embed_reduce_fn(embeddings_full, axis=axis), wt_norm # .sum()

        else:
            output = self.get_logits(sequence)
            s = output.logits.sequence.squeeze()[1:-1, :33].cpu()
            WTlogits = pd.DataFrame(s, columns=self.ordered_tokens, index=list(sequence)).T.iloc[4:29]
            WTlogits.columns = [j.split('.')[0] + ' ' + str(i + 1) for i, j in enumerate(WTlogits.columns)]
            wt_norm = np.diag(WTlogits.loc[[i.split(' ')[0] for i in WTlogits.columns]])
            LLR = WTlogits - wt_norm
            if return_embeddings:
                return output.embeddings.squeeze()[1:-1, :].cpu().numpy()
            else:
                return LLR, embed_reduce_fn(output.embeddings.squeeze()[1:-1, :].cpu().numpy(), axis=axis), wt_norm #.sum()
            

    def get_LLR(self, input_LL, aaref, pos, aaalt=None):
        if aaalt:
            llr = input_LL[pos - 1, self.tokenizers.sequence.vocab[aaalt]] - input_LL[pos - 1, self.tokenizers.sequence.vocab[aaref]]
            return llr.item()
        else:
            return {aa: (input_LL[pos - 1, idx] - input_LL[pos - 1, self.tokenizers.sequence.vocab[aaref]]).item() for aa, idx in self.aa_tokenizers.items()}

    def get_PLLR(self, wt_seq, mut_seq, start_pos=None, reduce=np.sum, weighted=False, device=0):
        if self.esmc_client:
            llr, ref_llr, ref_embeddings = self.get_ESMC_LL(wt_seq, return_embeddings=True)
            llr, mut_llr, mut_embeddings = self.get_ESMC_LL(mut_seq, return_embeddings=True)
            llr = mut_llr.sum() - ref_llr.sum()
            embeddings_dif = euclidean_distance(mut_embeddings.mean(axis=0), ref_embeddings.mean(axis=0))
            embeddings = [mut_embeddings, ref_embeddings]
            return llr, embeddings_dif, embeddings
        else:
            full_wt_seq = wt_seq
            full_mut_seq = mut_seq

            if max(len(wt_seq), len(mut_seq)) > self.max_len:
                wt_seq, mut_seq, start_pos = self.crop_variant(wt_seq, mut_seq, start_pos)
            wt_pll = self.get_PLL(wt_seq, reduce=reduce)
            mut_pll = self.get_PLL(mut_seq, reduce=reduce)

            #wt_llr, wt_embeddings, wt_pll_full = self.get_LL(wt_pll)
            #mut_llr, mut_embeddings, mut_pll_full = self.get_LL(mut_pll)
            return mut_pll['sequence'] - wt_pll['sequence'], euclidean_distance(mut_pll['embeddings'].mean(axis=0), wt_pll['embeddings'].mean(axis=0))#, mut_pll_full - wt_pll_full, euclidean_distance(wt_embeddings, mut_embeddings)

    def get_PLL(self, sequence, reduce=np.sum):
        s = self.get_logits(sequence)
        pll_dict = {}
        if self.logits_config.sequence:
            idx = [self.tokenizers.sequence.vocab[i] for i in sequence]
            pll_dict['sequence'] = np.sum(np.diag(s.logits.sequence.squeeze()[1:-1, idx].cpu()))
        if self.logits_config.return_embeddings:
            pll_dict['embeddings'] = s.embeddings.squeeze()[:, 1:-1].cpu()
        if self.logits_config.function:
            pll_dict['function'] = s.logits.function
        return pll_dict

    @torch.no_grad()
    def get_individual_embeddings(self, sequence):
        """
        Extract embeddings and features from different components of the ESM-3 model.
        
        Args:
            sequence (str): Input protein sequence
            
        Returns:
            dict: Dictionary containing model features and embeddings
        """
        features = {
            'encoder_embeddings': {},
            'transformer_features': {},
            'head_outputs': {},
            'attention_patterns': []
        }

        try:
            # Create protein tensor and set config
            protein = ESMProtein(sequence=sequence)
            self.logits_config.return_embeddings = True  # Ensure embeddings are returned
            
            # Get initial outputs
            protein_tensor = self.client.encode(protein)
            outputs = self.client.logits(protein_tensor, self.logits_config)
            
            if not hasattr(outputs, 'embeddings') or outputs.embeddings is None:
                raise ValueError("No embeddings returned from model")
            
            # Store embeddings
            features['encoder_embeddings']['sequence'] = outputs.embeddings.detach().cpu()
            
            # Process transformer layers
            hidden_states = outputs.embeddings
            for idx, block in enumerate(self.client.transformer.blocks):
                # Store transformer features
                features['transformer_features'][idx] = {}
                
                # Process attention
                with torch.no_grad():
                    # Get attention weights
                    qkv = block.attn.layernorm_qkv(hidden_states)
                    q, k, v = qkv.chunk(3, dim=-1)
                    
                    # Calculate attention scores
                    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
                    attn_weights = F.softmax(attn_weights, dim=-1)
                    
                    # Store attention patterns
                    features['attention_patterns'].append({
                        'layer': idx,
                        'weights': attn_weights.detach().cpu(),
                        'values': v.detach().cpu()
                    })
                    
                    # Update hidden states for next layer
                    attention_output = torch.matmul(attn_weights, v)
                    hidden_states = block.attn.out_proj(attention_output)
                    if hasattr(block, 'ffn'):
                        hidden_states = block.ffn(hidden_states)
            
            # Store head outputs
            if hasattr(outputs, 'logits'):
                head_mapping = {
                    'sequence': 'sequence',
                    'structure': 'structure',
                    'secondary_structure': 'ss8',
                    'sasa': 'sasa',
                    'function': 'function',
                    'residue_annotations': 'residue'
                }
                
                for attr, key in head_mapping.items():
                    if hasattr(outputs.logits, attr):
                        logit = getattr(outputs.logits, attr)
                        if logit is not None:
                            features['head_outputs'][key] = logit.detach().cpu()
                
        except Exception as e:
            print(f"Error processing sequence: {str(e)}")
            return None

        return features

    
    def analyze_embeddings(self, features):
        """
        Analyze the extracted embeddings and features.
        
        Args:
            features (dict): Output from get_individual_embeddings_v2
            
        Returns:
            dict: Analysis results for each component
        """
        analysis = {
            'encoder_analysis': {},
            'transformer_analysis': {},
            'head_analysis': {},
            'attention_analysis': {}
        }
        
        # Analyze encoder embeddings
        for name, embedding in features['encoder_embeddings'].items():
            analysis['encoder_analysis'][name] = {
                'shape': embedding.shape,
                'mean': embedding.mean().item(),
                'std': embedding.std().item(),
                'sparsity': (embedding == 0).float().mean().item()
            }
        
        # Analyze transformer features
        for feature_type, layers in features['transformer_features'].items():
            analysis['transformer_analysis'][feature_type] = {}
            for layer_name, feature in layers.items():
                analysis['transformer_analysis'][feature_type][layer_name] = {
                    'shape': feature.shape,
                    'mean': feature.mean().item(),
                    'std': feature.std().item(),
                    'sparsity': (feature == 0).float().mean().item()
                }
        
        # Analyze head outputs
        for name, output in features['head_outputs'].items():
            analysis['head_analysis'][name] = {
                'shape': output.shape,
                'mean': output.mean().item(),
                'std': output.std().item(),
                'sparsity': (output == 0).float().mean().item()
            }
        
        # Analyze attention patterns
        for idx, pattern in enumerate(features['attention_patterns']):
            analysis['attention_analysis'][f'layer_{idx}'] = {
                'weight_stats': {
                    'shape': pattern['weights'].shape,
                    'mean': pattern['weights'].mean().item(),
                    'std': pattern['weights'].std().item(),
                    'entropy': -(pattern['weights'] * torch.log(pattern['weights'] + 1e-10)).sum().item()
                },
                'value_stats': {
                    'shape': pattern['values'].shape,
                    'mean': pattern['values'].mean().item(),
                    'std': pattern['values'].std().item()
                }
            }
        
        return analysis

    @torch.no_grad()
    def function_decode(self, output):
        function_decoder = ESM3_function_decoder_v0("cuda")
        function_tokenizer = EsmFunctionTokenizer()

        p_none_threshold = 0.05
        log_p = F.log_softmax(output[:, 1:-1, :], dim=3).squeeze(0)

        log_p_nones = log_p[:, :, function_tokenizer.vocab_to_index["<none>"]]
        p_none = torch.exp(log_p_nones).mean(dim=1)
        where_none = p_none > p_none_threshold

        log_p[~where_none, :, function_tokenizer.vocab_to_index["<none>"]] = -torch.inf
        function_token_ids = torch.argmax(log_p, dim=2)
        function_token_ids[where_none, :] = function_tokenizer.vocab_to_index["<none>"]

        predicted_function = function_decoder.decode(
            function_token_ids,
            tokenizer=function_tokenizer,
            annotation_threshold=0.1,
            annotation_min_length=5,
            annotation_gap_merge_max=3,
        )
        return predicted_function

    def get_window_logits(self, sequence, position):
        # Calculate the start and end indices for the window
        start = max(0, position - 750)  # Center the window on the position
        end = min(len(sequence), position + 750)  # Ensure the end does not exceed the sequence length
        
        # Extract the windowed sequence
        window_sequence = sequence[start:end]
        
        # Get the logits for the windowed sequence
        output = self.get_logits(window_sequence)
        
        return output

    @torch.no_grad()
    def get_attention_weights(self, sequence, position):
        """Extract attention weights from all transformer layers for a specific position"""
        # Initial setup
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.client.encode(protein)
        device = next(self.client.parameters()).device
        
        attention_weights = []
        hidden_states = self.client.encoder(protein_tensor)  # Get initial embeddings
        
        # Iterate through transformer blocks sequentially
        for block_idx in range(48):
            block = self.client.transformer.blocks[block_idx]
            
            # Convert block parameters to float32
            for param in block.parameters():
                param.data = param.data.float()
            
            attn = block.attn
            
            # Get Q, K, V from current hidden states
            qkv = attn.layernorm_qkv(hidden_states)  # Use current layer's input
            q, k, v = qkv.chunk(3, dim=-1)
            
            # Reshape for multi-head attention
            batch_size, seq_len, hidden_size = q.shape
            head_size = hidden_size // 32
            
            q = q.view(batch_size, seq_len, 32, head_size).transpose(1, 2)
            k = k.view(batch_size, seq_len, 32, head_size).transpose(1, 2)
            v = v.view(batch_size, seq_len, 32, head_size).transpose(1, 2)
            
            # Calculate attention scores
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_size)
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # Store attention weights for this layer
            pos_weights = attn_weights[..., position, :]
            attention_weights.append(pos_weights.squeeze(0))
            
            # Update hidden states for next layer
            attention_output = torch.matmul(attn_weights, v)
            attention_output = attention_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, hidden_size
            )
            attention_output = attn.out_proj(attention_output)
            
            # Apply FFN
            hidden_states = block.ffn(attention_output + hidden_states)  # Add residual connection
        
        return torch.stack(attention_weights)

    def analyze_feature_importance(self, sequence, position):
        """
        Analyze the importance of different input features at a specific position.
        
        Args:
            sequence (str): Input protein sequence
            position (int): Position to analyze importance for (0-based)
            
        Returns:
            dict: Importance scores for each feature type
        """
        embeddings = {}
        importance_scores = {}
        
        # Get embeddings from different input features
        with torch.no_grad():
            # Get main sequence embeddings and logits
            protein = ESMProtein(sequence=sequence)
            protein_tensor = self.client.encode(protein)
            outputs = self.client.logits(protein_tensor, self.logits_config)
            
            # Store sequence embeddings
            embeddings['sequence'] = outputs.embeddings
            
            # Get attention-based embeddings
            attention_weights = self.get_attention_weights(sequence, position)
            embeddings['attention'] = attention_weights.mean(dim=0)  # Average across layers
            
            # Get structure predictions if enabled in config
            if self.logits_config.structure:
                embeddings['structure'] = outputs.logits.structure
            
            if self.logits_config.secondary_structure:
                embeddings['secondary_structure'] = outputs.logits.secondary_structure
            
            if self.logits_config.sasa:
                embeddings['sasa'] = outputs.logits.sasa
            
            if self.logits_config.function:
                embeddings['function'] = outputs.logits.function
            
            if self.logits_config.residue_annotations:
                embeddings['residue'] = outputs.logits.sequence
        
        # Calculate feature importance scores using L2 norms
        for feature_name, feature_embedding in embeddings.items():
            if feature_embedding is not None:
                # Handle different embedding shapes
                if isinstance(feature_embedding, torch.Tensor):  # Ensure it's a tensor
                    # Handle different embedding shapes
                    if len(feature_embedding.shape) == 3:  # [batch, seq_len, hidden_dim]
                        feature_embedding = feature_embedding.squeeze(0)
                    
                    if len(feature_embedding.shape) == 2:  # [seq_len, hidden_dim]
                        if position < feature_embedding.size(0):
                            # Use L2 norm of the embeddings at the mutation position
                            importance_scores[feature_name] = torch.norm(
                                feature_embedding[position], p=2
                            ).item()
                    else:  # Handle other shapes (e.g., attention weights)
                        importance_scores[feature_name] = torch.norm(
                            feature_embedding, p=2
                        ).item()
        
        # Add normalized attention scores
        if 'attention' in embeddings and isinstance(attention_weights, torch.Tensor):
            importance_scores['attention_head_mean'] = attention_weights.mean().item()
            importance_scores['attention_head_std'] = attention_weights.std().item()
            
            # Add per-layer attention statistics
            for layer_idx in range(len(attention_weights)):
                importance_scores[f'attention_layer_{layer_idx}'] = (
                    attention_weights[layer_idx].mean().item()
                )
        
        return importance_scores

    def visualize_attention(self, sequence, position, layer_idx=-1):
        """Visualize attention weights around a specific position"""
        attention_weights = self.get_attention_weights(sequence, position)
        
        # Get weights from specified layer (default: last layer)
        if layer_idx == -1:
            layer_idx = len(attention_weights) - 1
        layer_weights = attention_weights[layer_idx]
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(layer_weights.cpu().numpy(), 
                    cmap='viridis',
                    xticklabels=list(sequence),
                    yticklabels=range(layer_weights.shape[0]))
        
        plt.title(f'Attention weights around position {position}')
        plt.xlabel('Sequence position')
        plt.ylabel('Attention head')

    def compare_variant_effects(self, ref_sequence, mut_sequence, mutation_position):
        """
        Analyze and compare feature importance between reference and mutant sequences
        
        Args:
            ref_sequence (str): Reference protein sequence
            mut_sequence (str): Mutant protein sequence 
            mutation_position (int): Position of the mutation (0-based)
        
        Returns:
            dict: Contains comparative analysis results
        """
        results = {}
        
        # Get attention weights for both sequences
        ref_attention = self.get_attention_weights(ref_sequence, mutation_position)
        mut_attention = self.get_attention_weights(mut_sequence, mutation_position)
        
        # Calculate attention difference
        attention_diff = mut_attention - ref_attention
        
        # Analyze feature importance for both sequences
        ref_features = self.analyze_feature_importance(ref_sequence, mutation_position)
        mut_features = self.analyze_feature_importance(mut_sequence, mutation_position)
        
        # Calculate feature importance differences
        feature_diffs = {}
        for feature in ref_features:
            feature_diffs[feature] = mut_features[feature] - ref_features[feature]
        
        # Get window around mutation for context
        window_size = 10
        start = max(0, mutation_position - window_size)
        end = min(len(ref_sequence), mutation_position + window_size + 1)
        
        results = {
            'attention_weights': {
                'reference': ref_attention,
                'mutant': mut_attention,
                'difference': attention_diff
            },
            'feature_importance': {
                'reference': ref_features,
                'mutant': mut_features,
                'difference': feature_diffs
            },
            'sequence_context': {
                'reference': ref_sequence[start:end],
                'mutant': mut_sequence[start:end],
                'position': mutation_position - start
            }
        }
        
        return results

    def visualize_variant_comparison(self, comparison_results, save_path=None, threshold=0.02):
        """
        Visualize the comparison between reference and mutant sequences with detailed attention analysis
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Get sequence context and position
        ref_seq = comparison_results['sequence_context']['reference']
        mut_seq = comparison_results['sequence_context']['mutant']
        mut_pos = comparison_results['sequence_context']['position']
        
        # Create sequence position labels with special tokens
        seq_labels = ['<start>'] + [f"{i}:{aa}" for i, aa in enumerate(ref_seq)] + ['<end>']
        
        # Calculate global attention change metric
        attn_diff = comparison_results['attention_weights']['difference']
        mean_diff = attn_diff.mean(dim=0).cpu().numpy()
        global_change = np.abs(mean_diff).mean()
        
        # Create a function to set up the x-axis ticks
        def setup_ticks(ax, seq_len):
            # Create tick positions (including start and end tokens)
            tick_positions = np.arange(seq_len + 2)
            # Set tick positions and labels
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(seq_labels, rotation=45, ha='right')
            # Ensure all ticks are visible
            ax.tick_params(axis='x', which='major', labelsize=8)
            
        # Plot 1: Reference sequence attention
        ax1 = plt.subplot(3, 1, 1)
        ref_attn = comparison_results['attention_weights']['reference']
        mean_ref_attn = ref_attn.mean(dim=0).cpu().numpy()
        sns.heatmap(mean_ref_attn,
                    cmap='viridis',
                    xticklabels=False,  # We'll set these manually
                    yticklabels=False,
                    square=False,
                    ax=ax1)
        setup_ticks(ax1, len(ref_seq))
        plt.title(f'Reference Sequence Attention (mutation at {mut_pos}:{ref_seq[mut_pos]})', fontsize=12, pad=10)
        plt.axvline(x=mut_pos+1, color='red', linestyle='--', alpha=0.5, label=f'Mutation pos {mut_pos}')
        plt.xlabel('Sequence Position (idx:AA)', fontsize=10)
        plt.ylabel('Attention Head', fontsize=10)
        plt.legend()
        
        # Plot 2: Mutant sequence attention
        ax2 = plt.subplot(3, 1, 2)
        mut_attn = comparison_results['attention_weights']['mutant']
        mean_mut_attn = mut_attn.mean(dim=0).cpu().numpy()
        sns.heatmap(mean_mut_attn,
                    cmap='viridis',
                    xticklabels=False,  # We'll set these manually
                    yticklabels=False,
                    square=False,
                    ax=ax2)
        setup_ticks(ax2, len(mut_seq))
        plt.title(f'Mutant Sequence Attention (mutation at {mut_pos}:{mut_seq[mut_pos]})', fontsize=12, pad=10)
        plt.axvline(x=mut_pos+1, color='red', linestyle='--', alpha=0.5, label=f'Mutation pos {mut_pos}')
        plt.xlabel('Sequence Position (idx:AA)', fontsize=10)
        plt.ylabel('Attention Head', fontsize=10)
        plt.legend()
        
        # Plot 3: Attention differences
        ax3 = plt.subplot(3, 1, 3)
        significant_mask = np.abs(mean_diff) >= threshold
        highlighted_diff = np.where(significant_mask, mean_diff, 0)
        
        sns.heatmap(highlighted_diff,
                    center=0,
                    cmap='RdBu_r',
                    xticklabels=False,  # We'll set these manually
                    yticklabels=False,
                    square=False,
                    vmin=-max(abs(mean_diff.min()), abs(mean_diff.max())),
                    vmax=max(abs(mean_diff.min()), abs(mean_diff.max())),
                    ax=ax3)
        setup_ticks(ax3, len(ref_seq))
        plt.title(f'Significant Attention Changes (threshold = {threshold})\n'
                  f'Global Attention Change: {global_change:.4f}\n'
                  f'Mutation: {ref_seq[mut_pos]} → {mut_seq[mut_pos]} at position {mut_pos}', 
                  fontsize=12, pad=10)
        plt.axvline(x=mut_pos+1, color='black', linestyle='--', alpha=0.5, label=f'Mutation pos {mut_pos}')
        plt.xlabel('Sequence Position (idx:AA)', fontsize=10)
        plt.ylabel('Attention Head', fontsize=10)
        plt.legend()
        
        # Adjust layout with more space for x-labels
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, global_change

    @torch.no_grad()
    def get_attention_analysis(self, sequence, position=None, clustering=True):
        """
        Enhanced attention analysis incorporating Provis concepts
        
        Args:
            sequence (str): Input protein sequence
            position (int, optional): Specific position to analyze
            clustering (bool): Whether to perform hierarchical clustering on attention patterns
        
        Returns:
            dict: Multi-scale attention analysis results
        """
        analysis_results = {
            'local_attention': [],    # Position-specific patterns
            'global_attention': [],   # Full sequence patterns
            'hierarchical_patterns': None,  # Clustered patterns
            'contact_predictions': None,    # Predicted contacts
            'layer_contributions': []       # Per-layer importance
        }
        
        try:
            print("Starting attention analysis...")
            
            # Create protein tensor
            protein = ESMProtein(sequence=sequence)
            protein_tensor = self.client.encode(protein)
            print(f"Created protein tensor: {protein_tensor}")
            
            # Enable all logits for complete output
            self.logits_config = LogitsConfig(
                sequence=True,
                structure=True,
                secondary_structure=True,
                sasa=True,
                function=True,
                residue_annotations=True,
                return_embeddings=True
            )
            
            print("Running model forward pass...")
            # Get model outputs
            outputs = self.client.logits(protein_tensor, self.logits_config)
            
            if not hasattr(outputs, 'embeddings'):
                raise ValueError("Outputs object has no embeddings attribute")
            if outputs.embeddings is None:
                raise ValueError("Embeddings are None")
            
            print(f"Got embeddings with shape: {outputs.embeddings.shape}")
            
            # Get embeddings and ensure they're in float32
            embeddings = outputs.embeddings
            if isinstance(embeddings, torch.Tensor) and embeddings.dtype == torch.bfloat16:
                embeddings = embeddings.to(torch.float32)
            
            seq_length = len(sequence)
            print(f"Processing sequence of length: {seq_length}")
            
            # Track attention at different scales
            for block_idx, block in enumerate(self.client.transformer.blocks):
                print(f"Processing transformer block {block_idx}")
                
                # Convert block parameters to float32 if needed
                for param in block.parameters():
                    if param.dtype == torch.bfloat16:
                        param.data = param.data.to(torch.float32)
                
                # Get attention patterns
                qkv = block.attn.layernorm_qkv(embeddings)
                if isinstance(qkv, torch.Tensor) and qkv.dtype == torch.bfloat16:
                    qkv = qkv.to(torch.float32)
                
                q, k, v = qkv.chunk(3, dim=-1)
                print(f"QKV shapes: q={q.shape}, k={k.shape}, v={v.shape}")
                
                # Calculate attention scores
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
                attn_weights = F.softmax(attn_weights, dim=-1)
                print(f"Attention weights shape: {attn_weights.shape}")
                
                # Store global attention patterns
                analysis_results['global_attention'].append(
                    attn_weights.squeeze(0).cpu().numpy()
                )
                
                # Position-specific analysis if requested
                if position is not None:
                    if position >= seq_length:
                        print(f"Warning: position {position} is out of bounds for sequence length {seq_length}")
                    else:
                        pos_weights = attn_weights[..., position, :]
                        analysis_results['local_attention'].append(
                            pos_weights.squeeze(0).cpu().numpy()
                        )
                
                # Calculate layer contribution scores
                try:
                    layer_importance = self._calculate_layer_importance(
                        attn_weights, embeddings, block
                    )
                    analysis_results['layer_contributions'].append(layer_importance)
                except Exception as e:
                    print(f"Error calculating layer importance: {str(e)}")
                
                # Update embeddings for next layer
                attention_output = torch.matmul(attn_weights, v)
                attention_output = attention_output.transpose(1, 2).contiguous().view(
                    embeddings.shape
                )
                attention_output = block.attn.out_proj(attention_output)
                embeddings = block.ffn(attention_output + embeddings)
            
            print("Processing complete, performing final analyses...")
            
            # Perform hierarchical clustering if requested
            if clustering and len(analysis_results['global_attention']) > 0:
                try:
                    analysis_results['hierarchical_patterns'] = self._cluster_attention_patterns(
                        analysis_results['global_attention']
                    )
                    print("Clustering completed successfully")
                except Exception as e:
                    print(f"Error during clustering: {str(e)}")
            
            # Predict contacts using attention patterns
            if len(analysis_results['global_attention']) > 0:
                try:
                    analysis_results['contact_predictions'] = self._predict_contacts(
                        analysis_results['global_attention']
                    )
                    print("Contact prediction completed successfully")
                except Exception as e:
                    print(f"Error during contact prediction: {str(e)}")
            
            print("Analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            print(f"Error in attention analysis: {str(e)}")
            print("Full error traceback:")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            print("Analysis process finished")

    def _calculate_layer_importance(self, attn_weights, hidden_states, block):
        """Calculate importance scores for each layer based on attention entropy"""
        try:
            entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1)
            
            # Calculate gradient norm (if available)
            grad_norm = 0.0
            if hidden_states.grad is not None:
                grad_norm = torch.norm(hidden_states.grad).item()
            
            # Calculate attention sparsity
            sparsity = (attn_weights > 0.1).float().mean().item()
            
            return {
                'entropy': entropy.mean().item(),
                'gradient_norm': grad_norm,
                'attention_sparsity': sparsity,
                'max_attention': attn_weights.max().item(),
                'min_attention': attn_weights.min().item()
            }
        except Exception as e:
            print(f"Error in _calculate_layer_importance: {str(e)}")
            return {
                'entropy': 0.0,
                'gradient_norm': 0.0,
                'attention_sparsity': 0.0,
                'max_attention': 0.0,
                'min_attention': 0.0
            }

    def _cluster_attention_patterns(self, attention_patterns, n_clusters=10):
        """
        Perform hierarchical clustering on attention patterns
        
        Args:
            attention_patterns (list): List of attention weight matrices
            n_clusters (int): Number of clusters to form
            
        Returns:
            dict: Clustering results
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            import numpy as np
            
            print("Starting attention pattern clustering...")
            print(f"Number of attention patterns: {len(attention_patterns)}")
            
            # Stack all attention patterns
            stacked_patterns = np.stack(attention_patterns)
            print(f"Stacked patterns shape: {stacked_patterns.shape}")
            
            # Reshape for clustering (flatten each attention matrix)
            n_layers = stacked_patterns.shape[0]
            n_positions = stacked_patterns.shape[1] * stacked_patterns.shape[2]
            reshaped_patterns = stacked_patterns.reshape(n_layers, n_positions)
            print(f"Reshaped patterns shape: {reshaped_patterns.shape}")
            
            # Ensure we don't try to create more clusters than samples
            n_clusters = min(n_clusters, n_layers)
            
            # Perform clustering
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            clusters = clustering.fit_predict(reshaped_patterns)
            
            # Calculate additional clustering metrics
            cluster_sizes = np.bincount(clusters)
            cluster_means = np.zeros((n_clusters, n_positions))
            cluster_stds = np.zeros((n_clusters, n_positions))
            
            for i in range(n_clusters):
                if np.any(clusters == i):
                    cluster_patterns = reshaped_patterns[clusters == i]
                    cluster_means[i] = np.mean(cluster_patterns, axis=0)
                    cluster_stds[i] = np.std(cluster_patterns, axis=0)
            
            return {
                'clusters': clusters,
                'n_clusters': n_clusters,
                'cluster_sizes': cluster_sizes,
                'cluster_means': cluster_means.reshape(n_clusters, stacked_patterns.shape[1], -1),
                'cluster_stds': cluster_stds.reshape(n_clusters, stacked_patterns.shape[1], -1),
                'original_shape': stacked_patterns.shape
            }
            
        except Exception as e:
            print(f"Error in attention pattern clustering: {str(e)}")
            print("Full error traceback:")
            import traceback
            traceback.print_exc()
            return {
                'clusters': None,
                'n_clusters': 0,
                'cluster_sizes': None,
                'cluster_means': None,
                'cluster_stds': None,
                'error': str(e)
            }

    def _predict_contacts(self, attention_patterns, threshold=0.1):
        """Predict protein contacts using attention patterns"""
        # Average attention across heads and layers
        # mean_attention = np.mean([p.mean(axis=0) for p in attention_patterns], axis=0)
        mean_attention = np.mean(attention_patterns[-1], axis=0) # Using the last layer
        
        # Predict contacts where attention exceeds threshold
        contacts = mean_attention > threshold
        
        return {
            'contact_map': contacts,
            'contact_probability': mean_attention,
            'threshold': threshold
        }

    def visualize_attention_analysis(self, analysis_results, save_path=None):
        """
        Enhanced visualization incorporating Provis-style multi-scale views
        """
        if analysis_results is None:
            print("No analysis results to visualize")
            return None
        
        try:
            fig = plt.figure(figsize=(20, 15))
            
            # 1. Global attention patterns
            if analysis_results['global_attention']:
                ax1 = plt.subplot(3, 2, 1)
                mean_global_attention = np.mean([p for p in analysis_results['global_attention']], axis=0)
                sns.heatmap(mean_global_attention, ax=ax1, cmap='viridis')
                ax1.set_title('Global Attention Pattern')
            
            # 2. Layer contributions
            if analysis_results['layer_contributions']:
                ax2 = plt.subplot(3, 2, 2)
                layer_entropy = [l['entropy'] for l in analysis_results['layer_contributions']]
                plt.plot(layer_entropy)
                ax2.set_title('Layer-wise Attention Entropy')
                ax2.set_xlabel('Layer')
                ax2.set_ylabel('Entropy')
            
            # 3. Contact predictions
            if analysis_results['contact_predictions'] is not None:
                ax3 = plt.subplot(3, 2, 3)
                contact_map = analysis_results['contact_predictions']['contact_map']
                sns.heatmap(contact_map, ax=ax3, cmap='RdBu_r')
                ax3.set_title('Predicted Contacts')
            
            # 4. Hierarchical patterns
            if analysis_results['hierarchical_patterns'] is not None and \
               analysis_results['hierarchical_patterns']['clusters'] is not None:
                ax4 = plt.subplot(3, 2, 4)
                cluster_sizes = analysis_results['hierarchical_patterns']['cluster_sizes']
                plt.bar(range(len(cluster_sizes)), cluster_sizes)
                ax4.set_title('Attention Pattern Clusters')
                ax4.set_xlabel('Cluster ID')
                ax4.set_ylabel('Cluster Size')
            
            # 5. Local attention if available
            if analysis_results['local_attention']:
                ax5 = plt.subplot(3, 2, (5, 6))
                local_attention = np.mean([p for p in analysis_results['local_attention']], axis=0)
                plt.plot(local_attention)
                ax5.set_title('Position-specific Attention')
                ax5.set_xlabel('Sequence Position')
                ax5.set_ylabel('Attention Weight')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
    def measure_variant_effect(self, ref_seq, alt_seq, ref_start):
        """
        Measure variant effect by comparing embeddings to a reference distribution.
        
        Args:
            ref_seq (str): Reference protein sequence
            alt_seq (str): Alternative (mutant) protein sequence
            ref_start (int): Start position of the variant
            
        Returns:
            dict: Dictionary containing various distance metrics and analyses
        """
        results = {}
        
        try:
            # Get embeddings and attention weights for reference and alternative sequences
            E_ref = self.get_LL(ref_seq, return_embeddings=True)
            E_alt = self.get_LL(alt_seq, return_embeddings=True)
            
            E_ref_mean = E_ref.mean(axis=0)
            E_alt_mean = E_alt.mean(axis=0)
            
            # 1. Direct comparison of ref and mutated sequence
            results['direct'] = {
                'euclidean': euclidean_distance(E_ref_mean, E_alt_mean),
                'cosine': cosine_similarity(E_ref_mean, E_alt_mean)
            }
            
            # If reference distribution is set up
            if self.ref_embeddings is not None:
                # 2. Distance to Distribution approach
                results['distribution'] = {}
                
                # a. Nearest Neighbor Distance
                ref_distances, _ = self.knn_model.kneighbors(E_ref.reshape(len(E_ref), -1))
                alt_distances, _ = self.knn_model.kneighbors(E_alt.reshape(len(E_alt), -1))
                
                results['distribution']['nearest_neighbor'] = {
                    'ref_mean_distance': np.mean(ref_distances),
                    'alt_mean_distance': np.mean(alt_distances),
                    'distance_change': np.mean(alt_distances) - np.mean(ref_distances)
                }
                
                # b. Density Estimation
                ref_log_likelihood = self.gmm_model.score_samples(E_ref.reshape(len(E_ref), -1))
                alt_log_likelihood = self.gmm_model.score_samples(E_alt.reshape(len(E_alt), -1))
                
                results['distribution']['density_estimation'] = {
                    'ref_log_likelihood': np.mean(ref_log_likelihood),
                    'alt_log_likelihood': np.mean(alt_log_likelihood),
                    'log_likelihood_change': np.mean(alt_log_likelihood) - np.mean(ref_log_likelihood)
                }
                
                # 3. Compare to reference distribution centroid
                results['ref_distribution_centroid'] = {
                    'ref_distance': euclidean_distance(E_ref_mean, self.ref_centroid),
                    'alt_distance': euclidean_distance(E_alt_mean, self.ref_centroid),
                    'distance_change': (
                        euclidean_distance(E_alt_mean, self.ref_centroid) -
                        euclidean_distance(E_ref_mean, self.ref_centroid)
                    )
                }
            
        except Exception as e:
            print(f"Error in measure_variant_effect: {str(e)}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        return results

    def visualize_variant_effect(self, effect_results, save_path=None):
        """
        Visualize the variant effect analysis results.
        
        Args:
            effect_results (dict): Results from measure_variant_effect
            save_path (str, optional): Path to save the visualization
        """
        if 'error' in effect_results:
            print(f"Cannot visualize results due to error: {effect_results['error']}")
            return
        
        try:
            fig = plt.figure(figsize=(15, 10))
            
            # 1. Centroid distances
            if 'centroid' in effect_results:
                ax1 = plt.subplot(2, 2, 1)
                distances = effect_results['centroid']
                plt.bar(['Euclidean', 'Cosine'], 
                       [distances['euclidean'], distances['cosine']])
                ax1.set_title('Distance between Reference and Alternative Centroids')
                ax1.set_ylabel('Distance')
            
            # 2. Distribution analysis
            if 'distribution' in effect_results:
                ax2 = plt.subplot(2, 2, 2)
                nn_results = effect_results['distribution']['nearest_neighbor']
                plt.bar(['Reference', 'Alternative'], 
                       [nn_results['ref_mean_distance'], nn_results['alt_mean_distance']])
                ax2.set_title('Nearest Neighbor Distances')
                ax2.set_ylabel('Mean Distance to k-NN')
                
                # Add density estimation results
                ax3 = plt.subplot(2, 2, 3)
                density_results = effect_results['distribution']['density_estimation']
                plt.bar(['Reference', 'Alternative'], 
                       [density_results['ref_log_likelihood'], density_results['alt_log_likelihood']])
                ax3.set_title('Log-likelihood Scores')
                ax3.set_ylabel('Mean Log-likelihood')
            
            # 3. Per-residue analysis
            if 'per_residue' in effect_results:
                ax4 = plt.subplot(2, 2, 4)
                local_distances = effect_results['per_residue']['local_distances']
                plt.plot(range(len(local_distances)), local_distances, marker='o')
                ax4.set_title('Per-residue Embedding Distances')
                ax4.set_xlabel('Position (relative to variant)')
                ax4.set_ylabel('Euclidean Distance')
                
                # Add vertical line at variant position
                plt.axvline(x=len(local_distances)//2, color='r', linestyle='--', 
                           label='Variant Position')
                plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"Error in visualize_variant_effect: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def set_reference_distribution(self, sequences, n_components=5):
        """
        Set up reference distribution by computing embeddings and fitting models.
        
        Args:
            sequences (list): List of protein sequences for reference distribution
            n_components (int): Number of components for GMM
        """
        # try:
        #     # Get embeddings and attention weights for all sequences
        #     all_embeddings = []
        #     all_attention = []
            
        #     for seq in sequences:
        #         output = self.get_logits(seq)
        #         embeddings = output.embeddings.squeeze()[1:-1].cpu().numpy()
                
        #         # Get attention weights from last layer
        #         attn_weights = self._get_sequence_attention_weights(seq)
                
        #         all_embeddings.append(embeddings)
        #         all_attention.append(attn_weights)
            
        #     # Stack all embeddings
        #     self.ref_embeddings = np.vstack(all_embeddings)
        #     self.attention_weights = np.vstack(all_attention)
            
        #     # Calculate attention-weighted centroid
        #     self.ref_centroid = self._calculate_attention_weighted_mean(
        #         self.ref_embeddings, 
        #         self.attention_weights
        #     )
            
        #     # Fit k-NN model
        #     self.knn_model = NearestNeighbors(n_neighbors=min(5, len(sequences)))
        #     self.knn_model.fit(self.ref_embeddings)
            
        #     # Fit GMM model
        #     n_components = min(n_components, len(sequences) // 10)
        #     self.gmm_model = GaussianMixture(n_components=n_components)
        #     self.gmm_model.fit(self.ref_embeddings)
            
        #     return True
            
        # except Exception as e:
        #     print(f"Error in set_reference_distribution: {str(e)}")
        #     import traceback
        #     traceback.print_exc()
        #     return False

        self.knn_model = NearestNeighbors(n_neighbors=min(5, len(sequences)))
        self.knn_model.fit(self.ref_embeddings)
        n_components = min(n_components, len(sequences) // 10)
        self.gmm_model = GaussianMixture(n_components=n_components)
        self.gmm_model.fit(self.ref_embeddings)

        return True

    def _get_sequence_attention_weights(self, sequence):
        """Get attention weights from the last layer for a sequence."""
        try:
            # Create protein tensor with only sequence
            protein = ESMProtein(sequence=sequence)
            
            # Use the client's encode method
            protein_tensor = self.client.encode(protein)
            
            with torch.no_grad():
                # Convert model parameters to float32
                last_block = self.client.transformer.blocks[-1]
                for param in last_block.parameters():
                    if param.dtype == torch.bfloat16:
                        param.data = param.data.to(torch.float32)
                
                # Get outputs and convert to float32
                outputs = self.client.logits(
                    protein_tensor,
                    LogitsConfig(
                        sequence=True,
                        structure=False,
                        secondary_structure=False,
                        sasa=False,
                        function=False,
                        residue_annotations=False,
                        return_embeddings=True
                    )
                )
                
                # Convert hidden states to float32
                hidden_states = outputs.embeddings.to(torch.float32)
                
                # Convert layernorm parameters to float32
                for module in last_block.attn.layernorm_qkv.modules():
                    if hasattr(module, 'weight') and module.weight is not None:
                        module.weight.data = module.weight.data.to(torch.float32)
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.data = module.bias.data.to(torch.float32)
                
                # Get QKV and ensure float32
                qkv = last_block.attn.layernorm_qkv(hidden_states)
                qkv = qkv.to(torch.float32)
                q, k, v = qkv.chunk(3, dim=-1)
                
                # Calculate attention scores
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
                attn_weights = F.softmax(attn_weights, dim=-1)
                
                # Average across attention heads
                attn_weights = attn_weights.mean(dim=1).squeeze().cpu().numpy()
                
                if len(attn_weights.shape) == 1:
                    # Handle single sequence case
                    attn_weights = attn_weights.reshape(1, -1)
                
                return attn_weights[1:-1]  # Remove special tokens
                
        except Exception as e:
            print(f"Error in _get_sequence_attention_weights: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_attention_weighted_mean(self, embeddings, attention_weights=None):
        """Calculate attention-weighted mean of embeddings."""
        if attention_weights is None:
            attention_weights = np.ones(embeddings.shape[0])
        # Normalize attention weights
        normalized_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        
        # Calculate weighted mean
        weighted_mean = np.average(embeddings, weights=normalized_weights, axis=0)
        return weighted_mean

    @torch.no_grad()
    def iterative_structure_decode(self, sequence_tokens, output, decoder=None, temperature=1.0, top_k=5, max_iterations=5):
        """
        Iteratively decode structure tokens using beam search and temperature sampling.
        
        Args:
            sequence_tokens (torch.Tensor): Input sequence tokens [1, seq_len]
            output: Model output containing structure logits [1, seq_len, 4096]
            decoder: ESM3 structure decoder model
            temperature (float): Sampling temperature (higher = more diverse)
            top_k (int): Number of top candidates to consider
            max_iterations (int): Maximum number of refinement iterations
            
        Returns:
            dict: Contains final structure tokens and coordinates
        """
        if decoder is None:
            decoder = ESM3_structure_decoder_v0("cuda")
            
        batch_size, seq_length = sequence_tokens.shape
        
        # Initialize structure tokens with argmax predictions
        structure_tokens = torch.argmax(output.logits.structure, dim=-1)  # [1, seq_len]
        structure_tokens = (
            structure_tokens.where(sequence_tokens != 0, 4098)  # BOS
            .where(sequence_tokens != 2, 4097)  # EOS
            .where(sequence_tokens != 31, 4100)  # Chainbreak
        )
        
        # Iterative refinement
        for iteration in range(max_iterations):
            # Decode current structure tokens to coordinates
            structure_out = decoder.decode(
                structure_tokens,
                torch.ones_like(sequence_tokens),
                torch.zeros_like(sequence_tokens),
            )
            bb_coords = structure_out["bb_pred"]  # [1, seq_len, 4, 3]
            
            # Get new structure predictions based on current coordinates
            new_output = self.client.logits(
                sequence_tokens=sequence_tokens,
                structure_coords=bb_coords,
                per_res_plddt=torch.ones_like(sequence_tokens[:, 1:-1]).float(),
            )
            
            # Apply temperature to logits
            scaled_logits = new_output.logits.structure / temperature  # [1, seq_len, 4096]
            
            # Get top-k candidates for each position
            top_k_logits, top_k_tokens = torch.topk(scaled_logits, k=top_k, dim=-1)  # [1, seq_len, top_k]
            
            # Sample from top-k using softmax probabilities
            probs = F.softmax(top_k_logits, dim=-1)  # [1, seq_len, top_k]
            sampled_indices = torch.multinomial(
                probs.view(-1, top_k), 
                num_samples=1
            ).view(batch_size, seq_length)  # [1, seq_len]
            
            # Update structure tokens with sampled values
            new_structure_tokens = torch.gather(
                top_k_tokens, 
                dim=-1, 
                index=sampled_indices.unsqueeze(-1)
            ).squeeze(-1)  # [1, seq_len]
            
            # Keep special tokens unchanged
            structure_tokens = (
                new_structure_tokens.where(sequence_tokens != 0, 4098)  # BOS
                .where(sequence_tokens != 2, 4097)  # EOS
                .where(sequence_tokens != 31, 4100)  # Chainbreak
            )
            
            # Calculate PLDDT scores for quality assessment
            plddt_scores = torch.sigmoid(new_output.plddt_logits).mean(dim=-1)  # [1, seq_len]
            mean_plddt = plddt_scores.mean().item()
            
            print(f"Iteration {iteration + 1}, Mean pLDDT: {mean_plddt:.3f}")
            
        # Final decoding
        final_structure = decoder.decode(
            structure_tokens,
            torch.ones_like(sequence_tokens),
            torch.zeros_like(sequence_tokens),
        )
        
        return {
            "structure_tokens": structure_tokens,
            "bb_coords": final_structure["bb_pred"],
            "final_plddt": plddt_scores
        }

    @torch.no_grad()
    def iterative_function_decode(self, sequence_tokens, output, function_decoder=None, 
                                function_tokenizer=None, temperature=1.0, top_k=5, 
                                max_iterations=5, p_none_threshold=0.05):
        """
        Iteratively decode function predictions using beam search and confidence thresholds.
        
        Args:
            sequence_tokens (torch.Tensor): Input sequence tokens [1, seq_len]
            output: Model output containing function logits [1, seq_len, 8, 260]
            function_decoder: ESM3 function decoder model
            function_tokenizer: Function tokenizer
            temperature (float): Sampling temperature
            top_k (int): Number of top candidates to consider
            max_iterations (int): Maximum number of refinement iterations
            p_none_threshold (float): Threshold for no-function probability
            
        Returns:
            dict: Contains final function predictions and confidence scores
        """
        if function_decoder is None:
            function_decoder = ESM3_function_decoder_v0("cuda")
        if function_tokenizer is None:
            function_tokenizer = EsmFunctionTokenizer()
            
        batch_size, seq_length, num_heads, vocab_size = output.logits.function.shape
        
        # Initialize with base predictions
        # Get predictions for each head separately
        log_p = F.log_softmax(output.logits.function[:, 1:-1, :, :], dim=-1)  # [1, seq_len-2, 8, 260]
        function_predictions = []
        
        for iteration in range(max_iterations):
            # Temperature scaling
            scaled_logits = log_p / temperature
            
            # Get top-k candidates for each position and head
            top_k_logits, top_k_tokens = torch.topk(scaled_logits, k=top_k, dim=-1)  # [1, seq_len-2, 8, top_k]
            
            # Sample from top-k using softmax probabilities for each head
            probs = F.softmax(top_k_logits, dim=-1)  # [1, seq_len-2, 8, top_k]
            
            # Reshape for sampling
            batch_seq_heads = batch_size * (seq_length-2) * num_heads
            sampled_indices = torch.multinomial(
                probs.view(batch_seq_heads, top_k), 
                num_samples=1
            ).view(batch_size, seq_length-2, num_heads)  # [1, seq_len-2, 8]
            
            # Get sampled function tokens for each head
            function_token_ids = torch.gather(
                top_k_tokens, 
                dim=-1, 
                index=sampled_indices.unsqueeze(-1)
            ).squeeze(-1)  # [1, seq_len-2, 8]
            
            # Handle <none> predictions
            none_idx = function_tokenizer.vocab_to_index["<none>"]
            log_p_nones = log_p[..., none_idx]  # [1, seq_len-2, 8]
            p_none = torch.exp(log_p_nones)  # [1, seq_len-2, 8]
            where_none = p_none > p_none_threshold  # [1, seq_len-2, 8]
            
            # Apply <none> mask
            function_token_ids = function_token_ids.squeeze(0)  # [seq_len-2, 8]
            where_none = where_none.squeeze(0)  # [seq_len-2, 8]
            function_token_ids[where_none] = none_idx
            
            # Decode current predictions
            current_prediction = function_decoder.decode(
                function_token_ids,  # [seq_len-2, 8]
                tokenizer=function_tokenizer,
                annotation_threshold=0.1,
                annotation_min_length=5,
                annotation_gap_merge_max=3,
            )
            
            # Store prediction
            function_predictions.append({
                'tokens': function_token_ids,
                'interpro_preds': current_prediction["interpro_preds"],
                'function_keywords': current_prediction["function_keywords"],
                'confidence': torch.exp(scaled_logits).max(dim=-1)[0].mean().item()
            })
            
            print(f"Iteration {iteration + 1}, Confidence: {function_predictions[-1]['confidence']:.3f}")
            
            # Update log probabilities based on current prediction
            # This helps in refining predictions in subsequent iterations
            new_output = self.client.logits(
                sequence_tokens=sequence_tokens,
                function_tokens=F.pad(function_token_ids.unsqueeze(0), (0, 0, 1, 1), value=0))  # Pad sequence dimension
            log_p = F.log_softmax(new_output.logits.function[:, 1:-1, :, :], dim=-1)
        
        # Select best prediction based on confidence
        best_prediction_idx = max(range(len(function_predictions)), 
                                key=lambda i: function_predictions[i]['confidence'])
        best_prediction = function_predictions[best_prediction_idx]
        
        return {
            'best_prediction': best_prediction,
            'all_predictions': function_predictions,
            'final_confidence': best_prediction['confidence']
        }

    def ensemble_decode(self, sequence_tokens, num_samples=5, temperature_range=(0.5, 1.5)):
        """
        Perform ensemble decoding by combining multiple predictions with different temperatures.
        
        Args:
            sequence_tokens (torch.Tensor): Input sequence tokens
            num_samples (int): Number of ensemble samples
            temperature_range (tuple): Range of temperatures to sample from
            
        Returns:
            dict: Contains ensemble predictions for both structure and function
        """
        structure_predictions = []
        function_predictions = []
        
        for i in range(num_samples):
            # Sample temperature
            temperature = random.uniform(temperature_range[0], temperature_range[1])
            
            # Get base model output
            output = self.client.logits(sequence_tokens=sequence_tokens)
            
            # Get structure prediction
            structure_result = self.iterative_structure_decode(
                sequence_tokens, 
                output,
                temperature=temperature
            )
            structure_predictions.append(structure_result)
            
            # Get function prediction
            function_result = self.iterative_function_decode(
                sequence_tokens,
                output,
                temperature=temperature
            )
            function_predictions.append(function_result)
            
            print(f"Ensemble sample {i + 1}/{num_samples} complete")
            
        # Aggregate structure predictions
        ensemble_bb_coords = torch.stack(
            [pred["bb_coords"] for pred in structure_predictions]
        ).mean(dim=0)
        
        ensemble_plddt = torch.stack(
            [pred["final_plddt"] for pred in structure_predictions]
        ).mean(dim=0)
        
        # Aggregate function predictions
        # Count frequency of function keywords
        keyword_counts = {}
        for pred in function_predictions:
            for keyword in pred['best_prediction']['function_keywords']:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Filter keywords by frequency threshold
        ensemble_keywords = [
            kw for kw, count in keyword_counts.items() 
            if count >= num_samples * 0.5  # At least 50% agreement
        ]
        
        return {
            'structure': {
                'bb_coords': ensemble_bb_coords,
                'plddt': ensemble_plddt,
                'individual_predictions': structure_predictions
            },
            'function': {
                'keywords': ensemble_keywords,
                'individual_predictions': function_predictions
            }
        }


