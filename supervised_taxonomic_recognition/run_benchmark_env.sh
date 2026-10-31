# Source from run_benchmark_*.sh — applies HF / Torch cache dirs (same tree as clip_siglip/.env.example).
# Prefer a checked-out `clip_siglip/.env`; fallback to `.env.example`.
_CLIPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../clip_siglip" && pwd)"
if [[ -f "${_CLIPDIR}/.env" ]]; then
  # shellcheck source=/dev/null
  set -a && source "${_CLIPDIR}/.env" && set +a
elif [[ -f "${_CLIPDIR}/.env.example" ]]; then
  # shellcheck source=/dev/null
  set -a && source "${_CLIPDIR}/.env.example" && set +a
fi
