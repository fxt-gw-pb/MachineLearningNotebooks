#!/bin/zsh
set -e
NOTEBOOK_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$NOTEBOOK_ROOT"
if [[ -n "${NOTEBOOK_PYTHON:-}" ]]; then
  notebook_python="$NOTEBOOK_PYTHON"
elif [[ -x /opt/anaconda3/bin/python ]]; then
  notebook_python=/opt/anaconda3/bin/python
else
  notebook_python=python3
fi
"$notebook_python" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,13), "请使用 Python 3.11–3.13；可设置 NOTEBOOK_PYTHON 指定解释器。"'
if [[ ! -x .venv/bin/python ]]; then
  "$notebook_python" -m venv .venv
fi
if [[ ! -f .venv/.notebooks-ready ]] || ! cmp -s requirements.txt .venv/.notebooks-ready; then
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m ipykernel install --prefix "$NOTEBOOK_ROOT/.venv" --name python3 --display-name "机器学习笔记 (Python 3)"
  cp requirements.txt .venv/.notebooks-ready
fi
exec .venv/bin/python -m jupyterlab --notebook-dir="$NOTEBOOK_ROOT"
