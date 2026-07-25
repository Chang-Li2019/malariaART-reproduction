#!/bin/bash
# Queued giants run: waits for the small-gene run to finish, hard-gates on the
# chunk-reuse stitch validation, then runs full DMS on the 21 giants (>1500aa)
# with exact chunk-reuse. Launch with nohup; safe to leave unattended.
set -u
cd /mnt/Work1/changli/MalariaGen/malariaPLM/mutagenesis_rf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALARIAGEN_DATA_DIR=/mnt/Work1/changli/MalariaGen/malariaART/data
PY=/home/changli/miniconda3/envs/esm_env/bin/python
D=/mnt/Work1/changli/MalariaGen/malariaPLM/mutagenesis_rf_results
SMALL_PID="${1:-464117}"

echo "[queued $(date)] waiting for small-gene run (PID $SMALL_PID) to finish..."
while kill -0 "$SMALL_PID" 2>/dev/null; do sleep 120; done
echo "[queued $(date)] small run finished. Validating chunk-reuse stitching on GPU..."

if ! $PY validate_chunkreuse.py --gene PF3D7_1451200 --n_mut 20 \
        > "$D/chunkreuse_validation.log" 2>&1; then
    echo "[queued $(date)] CHUNK-REUSE VALIDATION FAILED -- aborting giants."
    grep -E "RESULT|max|Δ|Error" "$D/chunkreuse_validation.log" | tail -10
    exit 1
fi
grep -E "RESULT|max\|Δ\||chunks|pooling" "$D/chunkreuse_validation.log"
echo "[queued $(date)] stitching PASS. Launching giants (chunk-reuse, short->long)..."

$PY -u run_all.py --min_len 1501 --max_len 100000 --method chunkreuse
echo "[queued $(date)] GIANTS RUN COMPLETE."
