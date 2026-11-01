#!/usr/bin/env bash
set -Eeuo pipefail

release_id=${1:?Usage: deploy.sh RELEASE_ID}
[[ "$release_id" =~ ^[a-f0-9]{40}$ ]] || { echo 'Invalid release ID' >&2; exit 1; }
base=/opt/nutrisnap-ml
release=${2:-/opt/nutrisnap-ml/releases/$release_id}
[[ "$release" == "$base/releases/$release_id" ]] || { echo 'Invalid release path' >&2; exit 1; }
image="nutrisnap-ml:$release_id"

# Serialize CI and manual deployments on the server.
exec 9>"$base/.deploy.lock"
flock -w 600 9
test -s "$base/models/food_yolov8.onnx"
mkdir -p "$release/models"
cp "$base/models/food_yolov8.onnx" "$release/models/food_yolov8.onnx"
docker build --label "org.opencontainers.image.revision=$release_id" -t "$image" "$release"
previous=$(docker inspect nutrisnap-ml --format '{{.Config.Image}}' 2>/dev/null || true)

compose() {
  NUTRISNAP_IMAGE="$1" docker compose -p nutrisnap-ml -f "$release/deploy/compose.yaml" "${@:2}"
}
rollback() {
  echo 'Deployment failed; restoring previous image.' >&2
  docker logs --tail 50 nutrisnap-ml >&2 || true
  if [[ -n "$previous" ]]; then
    compose "$previous" up -d --wait --wait-timeout 120
  fi
}
trap rollback ERR
compose "$image" up -d --wait --wait-timeout 120
docker exec -i nutrisnap-ml python - <<'PY'
from pathlib import Path
import httpx

with httpx.Client(base_url="http://nutrisnap-ml:8000", timeout=60) as client:
    response = client.get("/classes")
    response.raise_for_status()
    assert len(response.json()["classes"]) == 38
    with Path("/app/src/api/static/samples/steak.jpg").open("rb") as image:
        response = client.post("/predict", files={"image": ("steak.jpg", image, "image/jpeg")})
    response.raise_for_status()
    assert any(item["food_name"] == "스테이크" for item in response.json()["detections"])
print("Model load and sample inference passed")
PY
trap - ERR

# Keep a stable compose entry point for manual administration.
cp "$release/deploy/compose.yaml" "$base/compose.yaml"
printf 'NUTRISNAP_IMAGE=%s\n' "$image" > "$base/.env"
printf '%s\n' "$release_id" > "$base/deployed-revision"
echo "Deployed $release_id"
