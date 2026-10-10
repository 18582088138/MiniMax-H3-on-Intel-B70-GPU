#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${H3_DATA_DIR:-$HOME/h3-data}"
IMAGE="${H3_IMAGE:-intel/llm-scaler-omni:0.2.0-b1}"
NAME=minimax-h3-b70-torch
mkdir -p "$BASE_DIR"/{models/{diffusion_models,text_encoders,vae,loras/h3},input,output,user}
if [[ ! -f "$BASE_DIR/extra_model_paths.yaml" ]]; then
    cp "$SCRIPT_DIR/extra_model_paths.yaml" "$BASE_DIR/extra_model_paths.yaml"
fi
for file in \
    diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
    text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
    vae/minimax_h3_video_vae_fp16.safetensors \
    vae/minimax_h3_audio_vae_fp32.safetensors; do
    [[ -s "$BASE_DIR/models/$file" ]] || { echo "Missing model: $BASE_DIR/models/$file" >&2; exit 2; }
done
[[ -d /dev/dri ]] || { echo '/dev/dri not found; check GPU driver.' >&2; exit 2; }
if docker container inspect "$NAME" >/dev/null 2>&1; then
    echo "Reusing $NAME; existing mounts and settings are unchanged."
    docker start "$NAME"
    exit 0
fi
docker run -d --name "$NAME" --restart=no --device=/dev/dri \
    -p 127.0.0.1:8188:8188 --shm-size=8g \
    -e OMNI_ATTN_BACKEND=torch \
    -e NO_PROXY=localhost,127.0.0.1 -e no_proxy=localhost,127.0.0.1 \
    -v "$BASE_DIR/models:/llm/models:ro" \
    -v "$BASE_DIR/input:/llm/input" -v "$BASE_DIR/output:/llm/output" \
    -v "$BASE_DIR/user:/llm/user" \
    -v "$BASE_DIR/extra_model_paths.yaml:/llm/ComfyUI/extra_model_paths.yaml:ro" \
    "$IMAGE" python /llm/ComfyUI/main.py \
    --listen 0.0.0.0 --port 8188 --reserve-vram 4 --enable-manager \
    --enable-dynamic-vram --input-directory /llm/input \
    --output-directory /llm/output --user-directory /llm/user
