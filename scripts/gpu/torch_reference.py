"""公式 PyTorch 実装（mamba_ssm）で推論し、照合用に float32 wav と速度を保存する。

公式 inference.py / inference_chunk.py と同じ処理を行う。違いは次の 4 点。
- 出力を 16bit FLAC ではなく float32 wav で保存する（量子化誤差を照合に混ぜないため）
- 速度計測のために同じ入力を繰り返し処理する
- 既定で TF32 を無効にする（--allow-tf32 で有効）
- 重みを Hub の最新版ではなく、--repo-dir の model.safetensors から読む（revision を固定するため）

公式リポジトリを --repo-dir に置いた状態で、その中のモジュールを import して使う。

    python scripts/gpu/torch_reference.py --repo-dir models/original \
        --input models/original/noisy_audio/mic_test2.wav --output-dir work/torch/full
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size-in-seconds", type=float, default=None)
    parser.add_argument("--hop-length-portion", type=float, default=0.5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dump-dir", type=Path, default=None, help="一括処理時の中間出力を npy で保存する")
    parser.add_argument("--allow-tf32", action="store_true", help="既定では TF32 を切って FP32 で照合する")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32

    sys.path.insert(0, str(args.repo_dir.resolve()))
    from models.generator_SEMamba_time_d4 import SEMamba
    from models.stfts import mag_phase_istft, mag_phase_stft
    from utils.util import load_config, pad_or_trim_to_match

    cfg = load_config(str(args.repo_dir / "config.json"))
    n_fft, hop_size, win_size = cfg["stft_cfg"]["n_fft"], cfg["stft_cfg"]["hop_size"], cfg["stft_cfg"]["win_size"]
    base_sr = cfg["stft_cfg"]["sampling_rate"]
    compress_factor = cfg["model_cfg"]["compress_factor"]

    def make_even(v: float) -> int:
        v = int(round(v))
        return v if v % 2 == 0 else v + 1

    device = torch.device("cuda")
    model = SEMamba(cfg).to(device)
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(args.repo_dir / "model.safetensors")))
    model.eval()
    relu = torch.nn.ReLU()

    dump = None

    def enhance_segment(wav: torch.Tensor, sr: int) -> torch.Tensor:
        nf, hs, ws = (make_even(v * sr // base_sr) for v in (n_fft, hop_size, win_size))
        mag, pha, _ = mag_phase_stft(wav, n_fft=nf, hop_size=hs, win_size=ws,
                                     compress_factor=compress_factor, center=True, addeps=False)
        amp_g, pha_g, _ = model(mag, pha)
        if dump is not None:
            for k, v in {"stft_mag": mag, "stft_pha": pha, "model_amp": amp_g, "model_pha": pha_g}.items():
                np.save(dump / f"{k}.npy", v.float().cpu().numpy())
        m = torch.expm1(relu(amp_g))
        zero_portion = torch.sum(m == 0, 1) / m.shape[1]
        amp_g[:, :, (zero_portion > 0.5)[0]] = 0
        audio = mag_phase_istft(amp_g, pha_g, nf, hs, ws, compress_factor)
        return pad_or_trim_to_match(wav.detach(), audio, pad_value=1e-8)

    def enhance_chunked(wav_all: torch.Tensor, sr: int) -> torch.Tensor:
        chunk_size = int(args.chunk_size_in_seconds * sr)
        hop_length = int(args.hop_length_portion * chunk_size)
        window = torch.clamp(torch.hann_window(chunk_size).to(device), min=1e-7)
        out = torch.zeros_like(wav_all)
        wsum = torch.zeros_like(wav_all)
        num_chunks = max(1, math.ceil((wav_all.shape[1] - chunk_size) / hop_length) + 1)
        for c in range(wav_all.shape[0]):
            wav = wav_all[c : c + 1]
            for i in range(num_chunks):
                seg = wav[:, i * hop_length : i * hop_length + chunk_size] if i < num_chunks - 1 else wav[:, -chunk_size:]
                audio = enhance_segment(seg, sr)
                w = window[: audio.shape[1]]
                if i < num_chunks - 1:
                    out[c : c + 1, i * hop_length : i * hop_length + chunk_size] += audio * w
                    wsum[c : c + 1, i * hop_length : i * hop_length + chunk_size] += w
                else:
                    out[c : c + 1, -chunk_size:] += audio * w
                    wsum[c : c + 1, -chunk_size:] += w
        return out / wsum

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for path in args.input:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.T.copy()).to(device)
            dump = None
            if args.dump_dir is not None and args.chunk_size_in_seconds is None:
                dump = args.dump_dir / path.stem
                dump.mkdir(parents=True, exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(args.repeat):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                enhanced = enhance_segment(wav, sr) if args.chunk_size_in_seconds is None else enhance_chunked(wav, sr)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            sf.write(args.output_dir / f"{path.stem}.wav", enhanced.cpu().numpy().T, sr, subtype="FLOAT")
            measured = times[1:] if len(times) > 1 else times
            duration = data.shape[0] / sr
            print(json.dumps({
                "file": path.name, "sr": sr, "duration_s": round(duration, 3),
                "infer_s_median": round(float(np.median(measured)), 4),
                "rtf": round(float(np.median(measured)) / duration, 4),
                "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "tf32": args.allow_tf32, "gpu": torch.cuda.get_device_name(),
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
