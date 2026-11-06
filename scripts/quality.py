"""マニフェストの音声を元音声（clean_filepath）と比べ、PESQ（広帯域）と STOI を計算する。

元音声そのものの条件（clean_filepath がない、または audio_filepath と同じ）は計算しない。

    python scripts/quality.py --manifest work/reverb/rt60_0600ms/manifest.jsonl \
        --output results/quality_rt60_0600ms.jsonl
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi

SAMPLE_RATE = 16000


def score(row: dict) -> dict:
    ref, sr_r = sf.read(row["clean_filepath"], dtype="float64")
    deg, sr_d = sf.read(row["audio_filepath"], dtype="float64")
    assert sr_r == sr_d == SAMPLE_RATE, (row["sample_id"], sr_r, sr_d)
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    try:
        pesq_wb = float(pesq(SAMPLE_RATE, ref, deg, "wb"))
    except Exception as e:  # 無音に近いなどで計算できない場合
        pesq_wb = None
        print(json.dumps({"sample_id": row["sample_id"], "pesq_error": str(e)}))
    return {"sample_id": row["sample_id"], "condition": row.get("condition"),
            "pesq_wb": pesq_wb, "stoi": float(stoi(ref, deg, SAMPLE_RATE, extended=False))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("clean_filepath") and r["clean_filepath"] != r["audio_filepath"]]
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(score, rows, chunksize=8))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    p = [r["pesq_wb"] for r in results if r["pesq_wb"] is not None]
    s = [r["stoi"] for r in results]
    print(json.dumps({"manifest": str(args.manifest), "files": len(results),
                      "pesq_wb_mean": round(float(np.mean(p)), 3) if p else None,
                      "stoi_mean": round(float(np.mean(s)), 4) if s else None}))


if __name__ == "__main__":
    main()
