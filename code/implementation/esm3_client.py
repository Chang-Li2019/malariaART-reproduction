import torch
from esm.models.esm3 import ESM3
from esm.utils.constants.models import ESM3_OPEN_SMALL


def get_esm3_client(device=None):
    """Load the frozen ESM-3 Open Small model on the given device (CPU or CUDA)."""
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)
