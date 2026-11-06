"""マニフェストの音声のパスを、RE-USE で処理した後の音声に差し替えたマニフェストを作る。

RE-USE の出力は `<sample_id>.wav` の名前で 1 つのディレクトリに置かれている前提。
元のマニフェストに `clean_filepath` がなければ（元音声そのものを処理した場合）、元の音声のパスを入れる。

    python scripts/derive_manifest.py --manifest work/reverb/rt60_0600ms/manifest.jsonl \
        --audio-dir work/reverb/rt60_0600ms/reuse --condition rt60_0600ms+reuse \
        --output work/reverb/rt60_0600ms/manifest_reuse.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = args.manifest.resolve().parent
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    out = []
    for row in rows:
        src = Path(row["audio_filepath"])
        if not src.is_absolute():
            src = base / src
        enhanced = (args.audio_dir / f"{src.stem}.wav").resolve()
        if not enhanced.exists():
            raise FileNotFoundError(enhanced)
        out.append({**row, "audio_filepath": str(enhanced), "input_filepath": str(src.resolve()),
                    "clean_filepath": row.get("clean_filepath", str(src.resolve())),
                    "condition": args.condition})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"condition": args.condition, "files": len(out), "output": str(args.output)}))


if __name__ == "__main__":
    main()
