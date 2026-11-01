#!/usr/bin/env bash
set -Eeuo pipefail

# Installed on the server and used as the deployment key's forced command.
# No arbitrary SSH commands, port forwarding, or interactive shell are allowed.
if [[ ! ${SSH_ORIGINAL_COMMAND:-} =~ ^deploy\ ([a-f0-9]{40})$ ]]; then
  echo 'Only deploy RELEASE_ID is allowed' >&2
  exit 1
fi
release_id=${BASH_REMATCH[1]}
release="/opt/nutrisnap-ml/releases/$release_id"
mkdir -p "$release"
tar --no-same-owner --no-same-permissions -xzf - -C "$release"
bash /opt/nutrisnap-ml/deploy.sh "$release_id" "$release"
