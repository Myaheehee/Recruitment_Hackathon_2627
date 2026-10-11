#!/usr/bin/env bash
# Create (or update) the training environment at ~/venvs/rl.
#
#   training/setup_venv.sh
#
# Needs uv (https://docs.astral.sh/uv/) and, for GPU training, an NVIDIA
# driver new enough for CUDA 13. Downloads about 3 GB the first time.
set -euo pipefail
cd "$(dirname "$0")/.."                   # repository root

VENV="${VENV:-$HOME/venvs/rl}"
if [ ! -x "$VENV/bin/python" ]; then
    uv venv "$VENV" --python 3.12        # 3.12 = the judging image's Python
fi
uv pip install --python "$VENV/bin/python" -r training/requirements.txt

"$VENV/bin/python" -c "import jax; print('JAX devices:', jax.devices())"
echo "Ready. Check it with:  $VENV/bin/python training/test_parity.py"
