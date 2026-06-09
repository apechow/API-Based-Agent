#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."  # cd into API-Based-Agent/

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

AGENT="CodeActAgent"

# IMPORTANT: Because Agent's prompt changes fairly often in the rapidly evolving codebase of OpenDevin
# We need to track the version of Agent in the evaluation to make sure results are comparable
AGENT_VERSION=v$(poetry run python -c "import agenthub; from opendevin.controller.agent import Agent; print(Agent.get_cls('$AGENT').VERSION)")
export log_file='log.log'

COMMAND="poetry run python evaluation/webarena/run_infer.py \
  --agent-cls CodeActAgent \
  --llm-config llm \
  -e SSH_PASSWORD='"hello"'
  --start_task_id 27 \
  --max-iterations 18 \
  --data-split validation \
  --max-chars 10000000 \
  --eval-num-workers 1 \
  --eval-note ${AGENT_VERSION}_${LEVELS}"

# Run the command
eval $COMMAND
