#!/usr/bin/env bash
set -euo pipefail
IMAGE="${H3_IMAGE:-intel/llm-scaler-omni:0.2.0-b1}"
docker run --rm --device=/dev/dri "$IMAGE" python -c '
import torch
print("torch=", torch.__version__)
print("available=", torch.xpu.is_available(), "count=", torch.xpu.device_count())
assert torch.xpu.is_available(), "XPU unavailable"
x = torch.ones((64,64), device="xpu")
y = x @ x
torch.xpu.synchronize()
value = y[0,0].item()
print("matmul=", value)
assert value == 64.0
'
