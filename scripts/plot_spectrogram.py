"""入力と強調後の音声をスペクトログラムで並べる。

    uv run python scripts/plot_spectrogram.py --out results/figures/x.png a.wav b.wav ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or [p.parent.name + "/" + p.name for p in args.inputs]

    fig, axes = plt.subplots(len(args.inputs), 1, figsize=(10, 3 * len(args.inputs)), sharex=True)
    for ax, path, label in zip(np.atleast_1d(axes), args.inputs, labels):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        ax.specgram(data[:, 0], NFFT=2048, Fs=sr, noverlap=1536, cmap="magma", vmin=-140, vmax=-20)
        ax.set_title(label)
        ax.set_ylabel("Hz")
    np.atleast_1d(axes)[-1].set_xlabel("s")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)


if __name__ == "__main__":
    main()
