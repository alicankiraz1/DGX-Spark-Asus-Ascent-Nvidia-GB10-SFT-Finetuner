#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/sft-finetuner-env"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
MAIN_SCRIPT="${SCRIPT_DIR}/SFTFinetuner.py"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }

separator() {
    echo "──────────────────────────────────────────────────────────────────"
}

separator
info "SFT Finetuner – Environment Setup"
separator

if ! command -v python3 &>/dev/null; then
    error "python3 not found. Please install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Detected Python ${PY_VERSION}"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    error "Python 3.10 or newer is required (found ${PY_VERSION})."
    exit 1
fi

separator
if [ -d "${VENV_DIR}" ]; then
    info "Virtual environment already exists at ${VENV_DIR}"
    info "Re-using existing environment."
else
    info "Creating virtual environment at ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
    ok "Virtual environment created."
fi

info "Activating virtual environment ..."
source "${VENV_DIR}/bin/activate"
ok "Activated: $(which python)"

separator
info "Upgrading pip, setuptools, wheel ..."
pip install --upgrade pip setuptools wheel --quiet
ok "pip upgraded."

separator
info "Removing any existing PyTorch installation ..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

info "Installing PyTorch nightly with CUDA 13.0 support ..."
info "  Index: https://download.pytorch.org/whl/nightly/cu130"
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130
ok "PyTorch nightly (cu130) installed."

separator
if [ ! -f "${REQ_FILE}" ]; then
    error "requirements.txt not found at ${REQ_FILE}"
    exit 1
fi

info "Installing dependencies from requirements.txt ..."
pip install -r "${REQ_FILE}"
ok "All dependencies installed."

separator
info "Verifying GPU and CUDA availability ..."

if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
    ok "nvidia-smi check passed."
else
    warn "nvidia-smi not found – skipping driver check."
fi

python3 -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version    : {torch.version.cuda}')
    print(f'  GPU device      : {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  GPU memory      : {mem:.1f} GB')
else:
    print('  WARNING: CUDA is not available. Training will be very slow on CPU.')
"
ok "Environment verification complete."

separator
if [ ! -f "${MAIN_SCRIPT}" ]; then
    error "SFTFinetuner.py not found at ${MAIN_SCRIPT}"
    exit 1
fi

info "Launching SFT Finetuner ..."
separator
echo ""
python3 "${MAIN_SCRIPT}"
