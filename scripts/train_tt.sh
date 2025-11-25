#!/usr/bin/env bash
set -euo pipefail

# Defaults
ENV_VARIANT="${ENV_VARIANT:-baseline}"       # baseline | stage2 | stage3 | path/to/yaml
AGENT_VARIANT="${AGENT_VARIANT:-baseline}"   # baseline | fast | wide | stage2 | stage3 | path/to/json
REWARD_VARIANT="${REWARD_VARIANT:-baseline}" # baseline | stage2 | stage3 | path/to/reward.py
HP_VARIANT="${HP_VARIANT:-baseline}"         # baseline | stage2 | stage3 | path/to/yaml

STEPS="${STEPS:-300000}"
LR="${LR:-}"          # empty means: don't override YAML
SEED="${SEED:-}"      # empty means: don't override YAML
LOGDIR="${LOGDIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_VARIANT="$2"; shift 2 ;;
    --agent) AGENT_VARIANT="$2"; shift 2 ;;
    --reward) REWARD_VARIANT="$2"; shift 2 ;;
    --hparams|--hparam|--hp) HP_VARIANT="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --logdir) LOGDIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--env baseline|stage2|stage3|path.yaml] [--agent baseline|fast|wide|stage2|stage3|path.json] [--reward baseline|stage2|stage3|path.py] [--hparams baseline|stage2|stage3|path.yaml] [--steps N] [--lr X] [--seed S] [--logdir DIR]"
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

case "$ENV_VARIANT" in
  baseline) ENV_CONFIG="configs/environment_params.yaml" ;;
  stage2)   ENV_CONFIG="configs/environment_params.stage2.yaml" ;;
  stage3)   ENV_CONFIG="configs/environment_params.stage3.yaml" ;;
  *)        ENV_CONFIG="$ENV_VARIANT" ;;
esac

case "$AGENT_VARIANT" in
  baseline) AGENT_CONFIG="configs/agent_params.json" ;;
  fast)     AGENT_CONFIG="configs/agent_params.fast.json" ;;
  wide)     AGENT_CONFIG="configs/agent_params.wide.json" ;;
  stage2)   AGENT_CONFIG="configs/agent_params.stage2.json" ;;
  stage3)   AGENT_CONFIG="configs/agent_params.stage3.json" ;;
  *)        AGENT_CONFIG="$AGENT_VARIANT" ;;
esac

REWARD="$REWARD_VARIANT"

case "$HP_VARIANT" in
  baseline) HP_CONFIG="configs/hyper_params.yaml" ;;
  stage2)   HP_CONFIG="configs/hyper_params.stage2.yaml" ;;
  stage3)   HP_CONFIG="configs/hyper_params.stage3.yaml" ;;
  *)        HP_CONFIG="$HP_VARIANT" ;;
esac

# Ensure simulator running (Apptainer on ICE, Docker locally)
if command -v apptainer >/dev/null 2>&1; then
  if ! apptainer instance list | grep -q "^deepracer"; then
    echo "[info] starting apptainer instance 'deepracer'..."
    ./scripts/start_deepracer.sh
  fi
else
  if ! docker ps --format '{{.Names}}' | grep -q '^deepracer$'; then
    echo "[info] starting docker container 'deepracer'..."
    ./scripts/start_deepracer.sh
  fi
fi

CMD=(python -m src.cli train
  --env-config "$ENV_CONFIG"
  --agent-config "$AGENT_CONFIG"
  --reward "$REWARD"
  --hparams-config "$HP_CONFIG"
  --steps "$STEPS"
)

# Optional overrides
[[ -n "$LR" ]] && CMD+=(--lr "$LR")
[[ -n "$SEED" ]] && CMD+=(--seed "$SEED")
[[ -n "$LOGDIR" ]] && CMD+=(--logdir "$LOGDIR")

echo "[info] running: ${CMD[*]}"
"${CMD[@]}"
