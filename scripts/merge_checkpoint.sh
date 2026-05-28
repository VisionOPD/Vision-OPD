#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_BASE_DIR="${PROJECT_ROOT}/checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65/"
BASE_DIR="${BASE_DIR:-${1:-${DEFAULT_BASE_DIR}}}"
BASE_DIR="${BASE_DIR%/}"
ACTOR_DIR="${BASE_DIR}/actor"

if [ ! -d "${ACTOR_DIR}" ]; then
  echo "Actor checkpoint directory not found: ${ACTOR_DIR}" >&2
  exit 1
fi

echo "Merging ${ACTOR_DIR} -> ${BASE_DIR}"

# Remove previously merged top-level files so verl can overwrite the target
# directory cleanly while keeping the original actor shards intact.
find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type f -print -delete

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${BASE_DIR}"

echo "Merge completed."
