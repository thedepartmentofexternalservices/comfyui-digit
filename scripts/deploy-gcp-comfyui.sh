#!/usr/bin/env bash
# Deploy comfyui-digit updates to all ComfyUI VMs on GCP.
#
# Requirements:
#   - gcloud CLI authenticated (gcloud auth login)
#   - Compute Engine API enabled
#   - SSH/IAP access to target instances
#
# Usage:
#   ./scripts/deploy-gcp-comfyui.sh
#   GCP_PROJECT=my-project INSTANCE_FILTER="labels.app=comfyui" ./scripts/deploy-gcp-comfyui.sh
#
# Environment variables:
#   GCP_PROJECT        GCP project ID (default: active gcloud config project)
#   INSTANCE_FILTER    gcloud instances list --filter value (default: name~'comfy')
#   DIGIT_NODE_DIR     Path to comfyui-digit inside each VM (default: ComfyUI/custom_nodes/comfyui-digit)
#   COMFYUI_SERVICE    systemd unit to restart (default: comfyui; set empty to skip restart)
#   GIT_REF            Git ref to checkout/pull (default: master)
#   USE_IAP            Set to 1 to tunnel SSH through IAP (default: 0)

set -euo pipefail

GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
INSTANCE_FILTER="${INSTANCE_FILTER:-name~'comfy'}"
DIGIT_NODE_DIR="${DIGIT_NODE_DIR:-ComfyUI/custom_nodes/comfyui-digit}"
COMFYUI_SERVICE="${COMFYUI_SERVICE:-comfyui}"
GIT_REF="${GIT_REF:-master}"
USE_IAP="${USE_IAP:-0}"

if [[ -z "${GCP_PROJECT}" || "${GCP_PROJECT}" == "(unset)" ]]; then
  echo "ERROR: Set GCP_PROJECT or run: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi

SSH_FLAGS=(--quiet)
if [[ "${USE_IAP}" == "1" ]]; then
  SSH_FLAGS+=(--tunnel-through-iap)
fi

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
if [[ ! -d "${DIGIT_NODE_DIR}/.git" ]]; then
  echo "ERROR: ${DIGIT_NODE_DIR} is not a git checkout on \$(hostname)" >&2
  exit 1
fi
cd "${DIGIT_NODE_DIR}"
git fetch origin
git checkout "${GIT_REF}"
git pull --ff-only origin "${GIT_REF}"
echo "Updated to: \$(git rev-parse --short HEAD) on \$(hostname)"
if [[ -n "${COMFYUI_SERVICE}" ]]; then
  if systemctl list-unit-files "${COMFYUI_SERVICE}.service" >/dev/null 2>&1; then
    sudo systemctl restart "${COMFYUI_SERVICE}"
    echo "Restarted ${COMFYUI_SERVICE} on \$(hostname)"
  else
    echo "WARN: ${COMFYUI_SERVICE}.service not found on \$(hostname); skipping restart" >&2
  fi
fi
EOF
)

mapfile -t INSTANCES < <(
  gcloud compute instances list \
    --project="${GCP_PROJECT}" \
    --filter="${INSTANCE_FILTER} AND status=RUNNING" \
    --format='csv[no-heading](name,zone)'
)

if [[ "${#INSTANCES[@]}" -eq 0 ]]; then
  echo "No running instances matched filter: ${INSTANCE_FILTER}" >&2
  exit 1
fi

echo "Deploying ${GIT_REF} to ${#INSTANCES[@]} instance(s) in project ${GCP_PROJECT}"
echo "Filter: ${INSTANCE_FILTER}"
echo

FAILED=0
for row in "${INSTANCES[@]}"; do
  IFS=',' read -r name zone <<< "${row}"
  echo "=== ${name} (${zone}) ==="
  if gcloud compute ssh "${name}" \
      --project="${GCP_PROJECT}" \
      --zone="${zone}" \
      "${SSH_FLAGS[@]}" \
      --command "${REMOTE_CMD}"; then
    echo "OK: ${name}"
  else
    echo "FAILED: ${name}" >&2
    FAILED=1
  fi
  echo
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "One or more instances failed." >&2
  exit 1
fi

echo "Deployment complete."
