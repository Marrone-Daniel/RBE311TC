#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SDK_DIR="$ROOT_DIR/third_party/pyorbbecsdk"

cd "$ROOT_DIR"
mkdir -p third_party

if [[ ! -d "$SDK_DIR/.git" ]]; then
  echo "pyorbbecsdk is not present at $SDK_DIR"
  echo "Clone it first. If github.com times out, use a working network/proxy/mirror:"
  echo "  git clone https://github.com/orbbec/pyorbbecsdk.git $SDK_DIR"
  exit 2
fi

cd "$SDK_DIR"
git checkout main

PYTHON_EXECUTABLE="$(cd "$ROOT_DIR" && uv run python -c 'import sys; print(sys.executable)')"
PYBIND11_DIR="$(uv run python -m pybind11 --cmakedir)"
mkdir -p build
cd build

echo "Building pyorbbecsdk with Python: $PYTHON_EXECUTABLE"
echo "Using pybind11_DIR: $PYBIND11_DIR"

cmake \
  -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  ..
make -j"$(nproc)"
make install

cd "$ROOT_DIR"
export PYTHONPATH="$SDK_DIR/install/lib:${PYTHONPATH:-}"
uv run python -c "import pyorbbecsdk; print('pyorbbecsdk ok')"

echo
echo "For this shell, keep using:"
echo "  export PYTHONPATH=$SDK_DIR/install/lib:\$PYTHONPATH"
