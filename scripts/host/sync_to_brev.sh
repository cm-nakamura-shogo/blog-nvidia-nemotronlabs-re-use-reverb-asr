#!/usr/bin/env bash
# スクリプトを Brev の ASR 用 workspace へ送る
set -euo pipefail
cd "$(dirname "$0")/../.."

WORKSPACE="${BREV_WORKSPACE:-cm-nakamura-asr-benchmark}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/re-use-reverb-asr}"

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
cp -R scripts "${STAGE}/"

brev exec "${WORKSPACE}" "mkdir -p ${REMOTE_DIR}"
brev copy "${STAGE}/" "${WORKSPACE}:${REMOTE_DIR}/"
