#!/usr/bin/env bash
# Run the DeepResearch annotation pipeline over the bundled question sets.
#
#   bash run_annotation.sh                       # all three sources
#   bash run_annotation.sh --source fractal_questions --limit 5
#
# Environment overrides:
#   PYTHON                    python interpreter to use (default: python3)
#   ANNOTATION_WORKERS        concurrent question workers (default: 3)
#   ANNOTATION_TIMEOUT        per-question timeout in seconds (default: 900)
#   ANNOTATION_MAX_LLM_CALLS  LLM-call budget per question (default: 30)
#   QWEN_AGENT_CONFIG         path to the config JSON (default: ./config.json)
#
# Secrets are read from config.local.json (git-ignored) or from the
# LLM_API_KEY / MODELSCOPE_API_KEY / OPENAI_API_KEY environment variables.
# The web search backend needs SERPER_API_KEY; without it the tool falls back
# to the rate-limited Bing RSS endpoint.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"

exec "$PYTHON" tests/examples/annotate_questions.py \
  --workers "${ANNOTATION_WORKERS:-3}" \
  --timeout-seconds "${ANNOTATION_TIMEOUT:-900}" \
  --save-trace \
  --max-llm-calls "${ANNOTATION_MAX_LLM_CALLS:-30}" \
  "$@"
