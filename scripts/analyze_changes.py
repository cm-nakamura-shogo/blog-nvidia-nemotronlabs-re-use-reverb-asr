"""RE-USE で認識が悪化したファイルを調べる。

- 誤りの種類（置換、削除、挿入）ごとの増減
- 3 つの ASR で同じファイルが悪化しているか
- 悪化したファイルが、「元音声＋RE-USE」の STOI が低いファイルに偏っているか
- 悪化の大きい例

    uv run python scripts/analyze_changes.py --results results
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ASRS = ["faster-whisper-large-v3", "parakeet-tdt-ctc-0.6b-ja", "reazonspeech-k2-v2"]
RT60S = ["clean", "rt60_0300ms", "rt60_0600ms", "rt60_1000ms"]
THRESHOLD = 0.05


def load(results: Path, cond: str, asr: str) -> dict[str, dict]:
    [path] = (results / "cer" / cond).glob(f"*-{asr}-bs1-run1.details.jsonl")
    return {r["sample_id"]: r for r in map(json.loads, path.open())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    stoi_clean_reuse = {r["sample_id"]: r["stoi"]
                        for r in map(json.loads, (args.results / "quality" / "clean+reuse.jsonl").open())}
    low_stoi = {s for s, v in stoi_clean_reuse.items() if v < np.percentile(list(stoi_clean_reuse.values()), 10)}

    print("## 悪化したファイルの誤りの内訳（悪化したファイルだけで合計、RE-USE前→後）")
    worse_sets: dict[tuple[str, str], set[str]] = {}
    for rt in RT60S:
        for asr in ASRS:
            before, after = load(args.results, rt, asr), load(args.results, f"{rt}+reuse", asr)
            worse = {s for s in before if after[s]["normalized_cer"] - before[s]["normalized_cer"] >= THRESHOLD}
            worse_sets[(rt, asr)] = worse
            delta = Counter()
            for s in worse:
                for k in ("substitutions", "deletions", "insertions"):
                    delta[k] += after[s]["normalized_edits"][k] - before[s]["normalized_edits"][k]
            rep = sum(after[s]["suspicious_repetition"] for s in worse)
            in_low = len(worse & low_stoi)
            print(f"{rt:<12} {asr:<26} 悪化{len(worse):3d}件  置換{delta['substitutions']:+4d} 削除{delta['deletions']:+4d} "
                  f"挿入{delta['insertions']:+4d}  繰り返し疑い{rep:2d}  低STOI(下位10%)に含まれる{in_low:2d}件")

    print("\n## 3つのASRで共通して悪化したファイル")
    for rt in RT60S:
        sets = [worse_sets[(rt, a)] for a in ASRS]
        all3 = set.intersection(*sets)
        any_ = set.union(*sets)
        print(f"{rt:<12} いずれかで悪化{len(any_):3d}件  3つとも悪化{len(all3):2d}件  "
              f"3つとも悪化のうち低STOI{len(all3 & low_stoi):2d}件")

    print("\n## 低STOI（元音声＋RE-USEで下位10%）のファイルでの、元音声→元音声＋RE-USE のCER平均")
    for asr in ASRS:
        before, after = load(args.results, "clean", asr), load(args.results, "clean+reuse", asr)
        b = np.mean([before[s]["normalized_cer"] for s in low_stoi])
        a = np.mean([after[s]["normalized_cer"] for s in low_stoi])
        rest_b = np.mean([before[s]["normalized_cer"] for s in before if s not in low_stoi])
        rest_a = np.mean([after[s]["normalized_cer"] for s in before if s not in low_stoi])
        print(f"{asr:<26} 低STOI {len(low_stoi)}件: {b * 100:5.2f}% → {a * 100:5.2f}%   それ以外: {rest_b * 100:5.2f}% → {rest_a * 100:5.2f}%")

    print("\n## 悪化の大きい例（RT60 1.0秒、Whisper）")
    before, after = load(args.results, "rt60_1000ms", ASRS[0]), load(args.results, "rt60_1000ms+reuse", ASRS[0])
    clean = load(args.results, "clean", ASRS[0])
    for s in sorted(worse_sets[("rt60_1000ms", ASRS[0])],
                    key=lambda s: after[s]["normalized_cer"] - before[s]["normalized_cer"], reverse=True)[: args.examples]:
        print(f"- {s}  STOI(元音声＋RE-USE)={stoi_clean_reuse[s]:.2f}  CER 元{clean[s]['normalized_cer']:.2f} "
              f"残響{before[s]['normalized_cer']:.2f} →RE-USE {after[s]['normalized_cer']:.2f}")
        print(f"  正解  : {before[s]['normalized_reference']}")
        print(f"  残響  : {before[s]['normalized_hypothesis']}")
        print(f"  RE-USE: {after[s]['normalized_hypothesis']}")


if __name__ == "__main__":
    main()
