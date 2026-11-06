#!/usr/bin/env bash
# Brev の ASR 用 workspace（cm-nakamura-asr-benchmark）で、残響の実験を通しで実行する
#
#   1. RE-USE の実行環境を用意する（初回のみ）
#   2. FLEURS ja test に残響を付ける（RT60 0.3, 0.6, 1.0 秒）
#   3. 元音声と残響付き音声に RE-USE をかける
#   4. PESQ と STOI を計算する
#   5. 3 つの ASR で認識し、CER を計算する
#
# 前提：このリポジトリが ~/re-use-reverb-asr、前回の ASR リポジトリが ~/workspace/asr-benchmark にある
set -euo pipefail
cd "$(dirname "$0")/.."
REUSE_DIR="$(pwd)"
ASR_DIR="${ASR_DIR:-/home/ubuntu/workspace/asr-benchmark}"
ASR_PY="${ASR_DIR}/work/.venv/bin/python"
SRC_MANIFEST="${ASR_DIR}/manifests/fleurs-ja_jp-test.jsonl"
OUT="${REUSE_DIR}/work/reverb"
RES="${REUSE_DIR}/results"
ASR_IDS=(faster-whisper-large-v3 parakeet-tdt-ctc-0.6b-ja reazonspeech-k2-v2)
STEPS="${STEPS:-setup reverb enhance quality asr cer}"
mkdir -p "${OUT}" "${RES}"

has_step() { [[ " ${STEPS} " == *" $1 "* ]]; }

if has_step setup; then
  [[ -f work/.venv-gpu/bin/activate ]] || bash scripts/gpu/setup_brev.sh
  source work/.venv-gpu/bin/activate
  ~/.local/bin/uv pip install pyroomacoustics pesq pystoi scipy
  deactivate
fi
source work/.venv-gpu/bin/activate

if has_step reverb; then
  python scripts/make_reverb.py --manifest "${SRC_MANIFEST}" \
    --rt60 0.3 0.6 1.0 --rirs-per-rt60 20 --output-dir "${OUT}" | tee "${RES}/reverb_summary.jsonl"
  # 元音声の条件も、同じ形のマニフェストにしておく
  mkdir -p "${OUT}/clean"
  python - "${SRC_MANIFEST}" "${OUT}/clean/manifest.jsonl" <<'EOF'
import json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
base = src.resolve().parent
with open(dst, "w", encoding="utf-8") as f:
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        p = Path(row["audio_filepath"])
        row["audio_filepath"] = str((p if p.is_absolute() else base / p).resolve())
        row["condition"] = "clean"
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
EOF
fi

CONDS=(clean rt60_0300ms rt60_0600ms rt60_1000ms)

if has_step enhance; then
  for c in "${CONDS[@]}"; do
    mapfile -t inputs < <(python -c "import json,sys;[print(json.loads(l)['audio_filepath']) for l in open(sys.argv[1]) if l.strip()]" "${OUT}/${c}/manifest.jsonl")
    python scripts/gpu/torch_reference.py --repo-dir models/original --input "${inputs[@]}" \
      --output-dir "${OUT}/${c}/reuse" --repeat 1 > "${RES}/enhance_${c}.jsonl"
    python scripts/derive_manifest.py --manifest "${OUT}/${c}/manifest.jsonl" \
      --audio-dir "${OUT}/${c}/reuse" --condition "${c}+reuse" --output "${OUT}/${c}/manifest_reuse.jsonl"
  done
fi

MANIFESTS=()
for c in "${CONDS[@]}"; do
  MANIFESTS+=("${OUT}/${c}/manifest.jsonl" "${OUT}/${c}/manifest_reuse.jsonl")
done
cond_name() { local m="$1"; local c; c="$(basename "$(dirname "$m")")"; [[ "$m" == *_reuse.jsonl ]] && echo "${c}+reuse" || echo "${c}"; }

if has_step quality; then
  for m in "${MANIFESTS[@]}"; do
    name="$(cond_name "$m")"
    [[ "${name}" == clean ]] && continue  # 元音声どうしは比べない
    python scripts/quality.py --manifest "$m" --output "${RES}/quality/${name}.jsonl" --workers 8
  done | tee "${RES}/quality_summary.jsonl"
fi
deactivate

if has_step asr; then
  cd "${ASR_DIR}"
  for m in "${MANIFESTS[@]}"; do
    name="$(cond_name "$m")"
    matrix="${OUT}/matrix_${name}.yaml"
    {
      echo "manifest: $m"
      echo "batch_sizes: [1]"
      echo "repeats: 1"
      echo "seed: 20260821"
      echo "configs:"
      echo "  - configs/faster-whisper-large-v3.yaml"
      echo "  - configs/parakeet-tdt-ctc-0.6b-ja.yaml"
      echo "  - configs/reazonspeech-k2-v2.yaml"
    } > "${matrix}"
    "${ASR_PY}" scripts/run_matrix.py --matrix "${matrix}" --only "${ASR_IDS[@]}" \
      --output-dir "${RES}/asr/${name}" --status-dir "${RES}/asr_status/${name}" --resume
  done
  cd "${REUSE_DIR}"
fi

if has_step cer; then
  cd "${ASR_DIR}"
  for m in "${MANIFESTS[@]}"; do
    name="$(cond_name "$m")"
    mkdir -p "${RES}/cer/${name}"
    for f in "${RES}/asr/${name}"/*.jsonl; do
      stem="$(basename "$f" .jsonl)"
      "${ASR_PY}" scripts/calculate_cer.py --input "$f" \
        --details "${RES}/cer/${name}/${stem}.details.jsonl" --summary "${RES}/cer/${name}/${stem}.summary.json"
    done
  done
  cd "${REUSE_DIR}"
fi
