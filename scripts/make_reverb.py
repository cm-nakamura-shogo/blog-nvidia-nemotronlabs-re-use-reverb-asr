"""FLEURS のマニフェストの音声に、会議室を想定した残響を付ける。

pyroomacoustics の鏡像法で直方体の部屋の RIR を作り、元音声に畳み込む。
RT60 ごとに RIR を一定数作っておき、発話ごとに種を固定して割り当てる。

畳み込んだ音声は、RIR の直接音の位置で時間を合わせ、元音声と同じ長さに切り、
発話区間の RMS を元音声にそろえる。PESQ と STOI で元音声と比べられるようにするため。

    python scripts/make_reverb.py --manifest <fleurs.jsonl> --rt60 0.3 0.6 1.0 \
        --output-dir work/reverb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import fftconvolve

SAMPLE_RATE = 16000
# 8〜10 人程度が入る会議室を想定
ROOM_DIM = (7.0, 5.0, 3.0)
TABLE_MARGIN = 1.0  # 壁から机の端までの距離


def sample_positions(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """会議机の上のマイクと、机を囲む座った話者の位置を返す。距離は 1〜3m。"""
    mic = np.array([ROOM_DIM[0] / 2 + rng.uniform(-0.5, 0.5),
                    ROOM_DIM[1] / 2 + rng.uniform(-0.3, 0.3), 0.75])
    while True:
        dist = rng.uniform(1.0, 3.0)
        angle = rng.uniform(0, 2 * np.pi)
        src = mic + np.array([dist * np.cos(angle), dist * np.sin(angle), 0.0])
        src[2] = 1.2  # 座った話者の口の高さ
        if (TABLE_MARGIN / 2 < src[0] < ROOM_DIM[0] - TABLE_MARGIN / 2
                and TABLE_MARGIN / 2 < src[1] < ROOM_DIM[1] - TABLE_MARGIN / 2):
            return mic, src


def make_rir(rt60: float, rng: np.random.Generator, design_rt60: float | None = None) -> tuple[np.ndarray, dict]:
    """design_rt60 は吸音率の計算に使う値。省略すると rt60 と同じ。"""
    design_rt60 = design_rt60 or rt60
    e_absorption, max_order = pra.inverse_sabine(design_rt60, ROOM_DIM)
    mic, src = sample_positions(rng)
    room = pra.ShoeBox(ROOM_DIM, fs=SAMPLE_RATE, materials=pra.Material(e_absorption),
                       max_order=max_order)
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float64)
    meta = {
        "rt60_target": rt60,
        "rt60_design": round(design_rt60, 4),
        "e_absorption": round(float(e_absorption), 4),
        "rt60_measured": float(pra.experimental.measure_rt60(rir, fs=SAMPLE_RATE)),
        "distance_m": float(np.linalg.norm(src - mic)),
        "max_order": int(max_order),
        "mic": mic.round(3).tolist(),
        "source": src.round(3).tolist(),
    }
    return rir, meta


def calibrate_design_rt60(rt60: float, seed: int, n_probe: int = 4, iterations: int = 4) -> float:
    """実測 RT60 の中央値が目標に近づくよう、吸音率の計算に使う RT60 を調整する。

    Sabine の式で吸音率を決めて鏡像法で RIR を作ると、実測の RT60 は目標より長くなる。
    目標と実測の比で設計値を縮める操作を数回繰り返す。
    """
    design = rt60
    for i in range(iterations):
        rng = np.random.default_rng([seed, int(round(rt60 * 1000)), 99, i])
        measured = np.median([make_rir(rt60, rng, design)[1]["rt60_measured"] for _ in range(n_probe)])
        design *= rt60 / measured
    return design


def active_rms(x: np.ndarray) -> float:
    """発話区間の RMS。20ms 区間のうち、最大から 40dB 以内の区間だけで計算する。"""
    frame = SAMPLE_RATE // 50
    n = len(x) // frame
    if n == 0:
        return float(np.sqrt(np.mean(x**2)) + 1e-12)
    e = np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1)
    active = e[e > e.max() * 1e-4]
    return float(np.sqrt(active.mean()) + 1e-12)


def reverberate(clean: np.ndarray, rir: np.ndarray) -> tuple[np.ndarray, float]:
    """残響付きの音声と、クリップ回避のために下げた量（dB、0 なら下げていない）を返す。"""
    wet = fftconvolve(clean, rir)
    offset = int(np.argmax(np.abs(rir)))  # 直接音の到達位置
    wet = wet[offset : offset + len(clean)]
    wet *= active_rms(clean) / active_rms(wet)
    peak = np.abs(wet).max()
    attenuation_db = 0.0
    if peak > 0.99:
        wet *= 0.99 / peak
        attenuation_db = float(20 * np.log10(peak / 0.99))
    return wet, attenuation_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rt60", type=float, nargs="+", default=[0.3, 0.6, 1.0])
    parser.add_argument("--rirs-per-rt60", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    base = args.manifest.resolve().parent
    if args.limit:
        rows = rows[: args.limit]

    for rt60 in args.rt60:
        tag = f"rt60_{int(round(rt60 * 1000)):04d}ms"
        out_dir = args.output_dir / tag
        (out_dir / "audio").mkdir(parents=True, exist_ok=True)
        design = calibrate_design_rt60(rt60, args.seed)
        rng = np.random.default_rng([args.seed, int(round(rt60 * 1000))])
        rirs = [make_rir(rt60, rng, design) for _ in range(args.rirs_per_rt60)]
        np.savez(out_dir / "rirs.npz", *[r for r, _ in rirs])
        (out_dir / "rirs.json").write_text(json.dumps([m for _, m in rirs], indent=2))

        assign_rng = np.random.default_rng([args.seed, int(round(rt60 * 1000)), 1])
        out_rows = []
        for row in rows:
            src_path = Path(row["audio_filepath"])
            if not src_path.is_absolute():
                src_path = base / src_path
            clean, sr = sf.read(src_path, dtype="float64")
            assert sr == SAMPLE_RATE and clean.ndim == 1, (src_path, sr, clean.shape)
            rir_index = int(assign_rng.integers(len(rirs)))
            wet, attenuation_db = reverberate(clean, rirs[rir_index][0])
            dst = out_dir / "audio" / f"{row['sample_id']}.wav"
            sf.write(dst, wet.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
            out_rows.append({**row, "audio_filepath": str(dst.resolve()),
                             "clean_filepath": str(src_path.resolve()),
                             "condition": tag, "rir_index": rir_index,
                             "clip_attenuation_db": round(attenuation_db, 2)})
        with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        measured = [m["rt60_measured"] for _, m in rirs]
        attenuated = sum(1 for r in out_rows if r["clip_attenuation_db"] > 0)
        print(json.dumps({"condition": tag, "files": len(out_rows), "clip_attenuated_files": attenuated,
                          "rt60_measured_median": round(float(np.median(measured)), 3),
                          "rt60_measured_range": [round(min(measured), 3), round(max(measured), 3)]}))


if __name__ == "__main__":
    main()
