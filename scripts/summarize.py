"""Brev から取得した結果を集計し、条件ごとの CER、PESQ と STOI、ファイルごとの悪化と改善の件数を表にする。

    uv run python scripts/summarize.py --results results --output results/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ASRS = {
    "faster-whisper-large-v3": "Whisper large-v3",
    "parakeet-tdt-ctc-0.6b-ja": "Parakeet TDT-CTC 0.6B",
    "reazonspeech-k2-v2": "ReazonSpeech K2 v2",
}
RT60S = ["clean", "rt60_0300ms", "rt60_0600ms", "rt60_1000ms"]
LABELS = {"clean": "元音声", "rt60_0300ms": "RT60 0.3秒", "rt60_0600ms": "RT60 0.6秒", "rt60_1000ms": "RT60 1.0秒"}
CHANGE_THRESHOLD = 0.05  # ファイルごとの悪化と改善を数えるしきい値（正規化 CER で 5 ポイント）


def label(cond: str) -> str:
    base = cond.removesuffix("+reuse")
    return LABELS[base] + ("＋RE-USE" if cond.endswith("+reuse") else "")


def load_details(results: Path, cond: str, asr: str) -> dict[str, dict]:
    [path] = (results / "cer" / cond).glob(f"*-{asr}-bs1-run1.details.jsonl")
    return {r["sample_id"]: r for r in map(json.loads, path.open())}


def load_summary(results: Path, cond: str, asr: str) -> dict:
    [path] = (results / "cer" / cond).glob(f"*-{asr}-bs1-run1.summary.json")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/summary.md"))
    args = parser.parse_args()
    conds = [c for rt in RT60S for c in (rt, f"{rt}+reuse")]
    lines = []

    lines += ["## 正規化CER", "", "| 条件 | " + " | ".join(ASRS.values()) + " |",
              "| --- | " + " | ".join("---:" for _ in ASRS) + " |"]
    cer = {}
    for c in conds:
        cer[c] = {a: load_summary(args.results, c, a)["normalized_cer"] for a in ASRS}
        lines.append(f"| {label(c)} | " + " | ".join(f"{cer[c][a] * 100:.2f}%" for a in ASRS) + " |")
    (args.results / "cer_summary.json").write_text(json.dumps(cer, indent=2))

    lines += ["", "## PESQ（広帯域）とSTOI（元音声との比較）", "",
              "| 条件 | PESQ 平均 | STOI 平均 | STOI 10パーセンタイル |", "| --- | ---: | ---: | ---: |"]
    for c in conds[1:]:
        rows = [json.loads(l) for l in (args.results / "quality" / f"{c}.jsonl").open()]
        p = np.array([r["pesq_wb"] for r in rows if r["pesq_wb"] is not None])
        s = np.array([r["stoi"] for r in rows])
        lines.append(f"| {label(c)} | {p.mean():.2f} | {s.mean():.3f} | {np.percentile(s, 10):.3f} |")

    lines += ["", f"## ファイルごとの悪化と改善（正規化CERが{int(CHANGE_THRESHOLD * 100)}ポイント以上動いた件数、悪化/改善）", "",
              "| ASR | " + " | ".join(LABELS[rt] for rt in RT60S) + " |", "| --- | " + " | ".join("---" for _ in RT60S) + " |"]
    for a, name in ASRS.items():
        cells = []
        for rt in RT60S:
            before, after = load_details(args.results, rt, a), load_details(args.results, f"{rt}+reuse", a)
            diff = [after[s]["normalized_cer"] - before[s]["normalized_cer"] for s in before]
            cells.append(f"{sum(d >= CHANGE_THRESHOLD for d in diff)}/{sum(d <= -CHANGE_THRESHOLD for d in diff)}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
