# shellcheck source=/dev/null
_CLIPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../clip_siglip" && pwd)"
if [[ -f "${_CLIPDIR}/.env" ]]; then set -a && source "${_CLIPDIR}/.env" && set +a
elif [[ -f "${_CLIPDIR}/.env.example" ]]; then set -a && source "${_CLIPDIR}/.env.example" && set +a
fi

# Default epoch count for train_seg.py wrappers (`export SEG_EPOCHS=25` to override).
: "${SEG_EPOCHS:=100}"
