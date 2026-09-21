#!/usr/bin/env bash
# Phase 0: 環境構築スクリプト（Apple Silicon / MPS 前提）
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v /opt/homebrew/bin/python3.11 >/dev/null 2>&1; then
  echo "python3.11 が見つかりません。'brew install python@3.11' を先に実行してください。"
  exit 1
fi

if [ ! -d .venv ]; then
  /opt/homebrew/bin/python3.11 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== MPS availability check ==="
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

echo "セットアップ完了。以後は 'source .venv/bin/activate' でこの環境に入れます。"
