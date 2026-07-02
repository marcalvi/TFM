#!/usr/bin/env bash
# M3TRICS Dashboard Launcher
# Run after completing analysis:
#   bash run_dashboard.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║        M3TRICS Interactive Dashboard         ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Python interpreter ──────────────────────────────────────────────────────
# Override with: PYTHON=/path/to/python bash run_dashboard.sh
if [[ -n "${PYTHON:-}" ]]; then
    :
elif command -v python &>/dev/null; then
    PYTHON=python
elif [[ -x "/opt/miniconda3/bin/python" ]]; then
    PYTHON=/opt/miniconda3/bin/python
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    echo "ERROR: Python not found. Activate your conda environment first." >&2
    exit 1
fi

# ── Configuration ───────────────────────────────────────────────────────────
# Results directory. Override if needed.
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
if [[ "${RESULTS_DIR}" != /* ]]; then
    RESULTS_DIR="${SCRIPT_DIR}/${RESULTS_DIR}"
fi

# Optional fallback analysis directory for legacy dashboards. Override if needed.
ANALYSIS_DIR="${ANALYSIS_DIR:-${SCRIPT_DIR}/analysis/progressive_missingness_analysis_outputs}"
if [[ "${ANALYSIS_DIR}" != /* ]]; then
    ANALYSIS_DIR="${SCRIPT_DIR}/${ANALYSIS_DIR}"
fi

# Models starting with DI- are detected automatically; legacy _KD outputs are also supported.
# Optionally pass custom/legacy distillation model names as a comma-separated list.
DISTILLATION_MODELS="${DISTILLATION_MODELS:-}"

# Output HTML file
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/dashboard/m3trics_dashboard.html}"

# ── Generate dashboard ──────────────────────────────────────────────────────
cd "${REPO_ROOT}"
${PYTHON} "${SCRIPT_DIR}/dashboard/generate_dashboard.py" \
    --results_dir         "${RESULTS_DIR}" \
    --analysis_dir        "${ANALYSIS_DIR}" \
    --distillation_models "${DISTILLATION_MODELS}" \
    --output              "${OUTPUT}"

# ── Open in browser ─────────────────────────────────────────────────────────
if [[ -f "${OUTPUT}" ]]; then
    echo ""
    if [[ "$(uname)" == "Darwin" ]]; then
        open "${OUTPUT}"
    elif command -v xdg-open &>/dev/null; then
        xdg-open "${OUTPUT}"
    else
        echo "Open manually: ${OUTPUT}"
    fi
fi
