# 会議室の残響をRE-USEで取り除き、日本語音声認識の精度を比べる

FLEURS ja test（650件）に会議室を想定した残響を付け、NVIDIA NemotronLabs RE-USEで処理した前後で、日本語の音声認識の精度を比べる検証のコードと結果です。

> [!NOTE]
> 発話ごとの認識結果と、ファイルごとの指標は含めていません。集計結果と図だけを置いています。

## 結果

正規化CER（BS1、greedy）です。

| 条件 | Whisper large-v3 | Parakeet TDT-CTC 0.6B | ReazonSpeech K2 v2 |
| --- | ---: | ---: | ---: |
| 元音声 | 4.49% | 5.94% | 9.40% |
| 元音声＋RE-USE | 4.48% | 5.76% | 8.29% |
| RT60 0.3秒 | 4.92% | 6.03% | 9.99% |
| RT60 0.3秒＋RE-USE | 4.57% | 5.71% | 8.37% |
| RT60 0.6秒 | 5.91% | 6.53% | 10.64% |
| RT60 0.6秒＋RE-USE | 4.88% | 5.81% | 8.45% |
| RT60 1.0秒 | 8.05% | 7.74% | 12.69% |
| RT60 1.0秒＋RE-USE | 5.07% | 6.01% | 8.65% |

PESQ、STOI、ファイルごとの悪化と改善の件数は [`results/summary.md`](results/summary.md) にあります。

## 実験の条件

- 音声：FLEURS ja_jp test 650件（16kHz）
- 残響：pyroomacousticsの鏡像法で作ったRIR
  - 部屋は7×5×3m。マイクは机の上（高さ0.75m）、話者は座った口の高さ（1.2m）でマイクから1〜3m
  - RT60は0.3、0.6、1.0秒。RT60ごとにRIRを20本作り、発話ごとに種を固定して割り当てる
  - Sabineの式のままでは実測RT60が目標の約1.5倍になるため、設計値を較正して実測を目標に合わせた（実測の中央値は0.307、0.602、0.995秒）
  - 直接音の位置で時間を合わせ、長さと発話区間のRMSを元音声にそろえた
- 音声強調：`nvidia/NVIDIA-NemotronLabs-RE-USE`（revision `31f82f2`）の公式実装をFP32で実行
- 音声認識：ASR比較の記事の[japanese-asr-l40s-benchmark](https://github.com/cm-nakamura-shogo/japanese-asr-l40s-benchmark)の設定とスクリプトをそのまま使用
- 実行環境：NVIDIA Brev、L40S 1基

## ディレクトリ

```text
scripts/
├── make_reverb.py       # FLEURSに残響を付ける（RT60の較正つき）
├── derive_manifest.py   # RE-USEで処理した音声のマニフェストを作る
├── quality.py           # 元音声と比べてPESQとSTOIを計算する
├── run_pipeline.sh      # Brevで残響付与からCERの計算までを通しで実行する
├── summarize.py         # 結果を集計してresults/summary.mdを作る
├── analyze_changes.py   # RE-USEで認識が悪化したファイルを調べる
├── plot_spectrogram.py  # スペクトログラムを並べた図を作る
├── gpu/                 # RE-USEの実行環境の構築と推論
└── host/                # Brevへの転送
results/
├── summary.md, cer_summary.json, quality_summary.jsonl
├── reverb_summary.jsonl, rirs/   # 作ったRIRの条件と実測RT60
├── enhance_*.jsonl               # RE-USEの処理時間
└── figures/
```

## 再現手順

前回の記事のASR環境（FLEURSとASRの仮想環境）があるBrevのworkspaceで実行します。

```bash
bash scripts/host/sync_to_brev.sh
brev exec cm-nakamura-asr-benchmark "cd ~/re-use-reverb-asr && bash scripts/run_pipeline.sh"
```

結果を取得したら、手元で集計します。

```bash
uv run python scripts/summarize.py
```

## 関連記事

- [NVIDIAの音声強調モデル「NemotronLabs RE-USE」を公式実装で動かしてみた（9.6Mパラメータで商用利用可）](https://dev.classmethod.jp/articles/nvidia-nemotronlabs-re-use-overview/)（[リポジトリ](https://github.com/cm-nakamura-shogo/blog-nvidia-nemotronlabs-re-use)）
- [NVIDIA L40Sで日本語対応ASRモデルを比較してみた](https://dev.classmethod.jp/articles/japanese-asr-l40s-fleurs-benchmark/)（[リポジトリ](https://github.com/cm-nakamura-shogo/japanese-asr-l40s-benchmark)）

## ライセンス

コードはMITライセンスです。FLEURSはCC BY 4.0、RE-USEのモデルはOpenMDW-1.1で提供されています。
