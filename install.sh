#!/usr/bin/env bash
# Install the EPR PC application into a local virtual environment.
#   ./install.sh          acquisition and tests
#   ./install.sh --ml     plus the recognition stack (PyTorch from PyPI)
set -euo pipefail
cd "$(dirname "$0")"

ML=0
for arg in "$@"; do
  case "$arg" in
    --ml) ML=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating virtual environment (.venv)"
  if command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv .venv
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m venv .venv
  else
    echo "Python 3.11 or newer is required." >&2
    exit 1
  fi
fi

.venv/bin/python -m pip install --upgrade pip
if [[ "$ML" -eq 1 ]]; then
  .venv/bin/python -m pip install -e ".[dev,ml]"
else
  .venv/bin/python -m pip install -e ".[dev]"
fi

echo
.venv/bin/epr --help
echo
echo "Activate with:  source .venv/bin/activate"
echo "Then run:       epr acquire"
if [[ "$ML" -eq 0 ]]; then
  echo "Recognition (CNN/LNN) needs:  ./install.sh --ml"
fi
