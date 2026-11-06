#!/usr/bin/env bash
# Brev（L40S, driver 565, Python 3.10）に公式 RE-USE の実行環境を作る。
# mamba_ssm と causal-conv1d はソースビルドを避け、公式リリースの wheel を使う。
# mamba_ssm 2.3 系は triton>=3.5 を要求して torch 2.6 と両立しないため、2.2.4 を使う。
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/0.13.4/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

rm -rf work/.venv-gpu
# triton が実行時に C 拡張をビルドするので、ヘッダ同梱の uv 管理 Python を使う（システムの 3.10 には Python.h がない）
uv venv --managed-python --python 3.10 work/.venv-gpu
source work/.venv-gpu/bin/activate

# 1 回の依存解決にまとめる。分けると後段で torch が cu130 版（driver 565 では動かない）に上がる
uv pip install \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match \
  "torch==2.6.0+cu124" "torchaudio==2.6.0+cu124" \
  "causal_conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
  "mamba_ssm @ https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
  einops safetensors soundfile numpy pyyaml librosa resampy "huggingface_hub<1" "transformers<5"

# 公式リポジトリ一式（推論コード、重み、サンプル）を revision 固定で取得する
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/NVIDIA-NemotronLabs-RE-USE",
                  revision="31f82f2efad1095034492f236e8dac9239ba713b",
                  local_dir="models/original")
EOF
sha256sum models/original/model.safetensors

python - <<'EOF'
import torch, mamba_ssm, causal_conv1d
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name())
print("mamba_ssm", mamba_ssm.__version__, "causal_conv1d", causal_conv1d.__version__)
EOF
