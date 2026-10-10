"""Sample device VRAM and container RAM without modifying inference."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--container', default='minimax-h3-b70-torch')
p.add_argument('--device', default='0', help='B70 PCI BDF; confirm with xpu-smi discovery')
p.add_argument('--output', required=True)
p.add_argument('--interval', type=float, default=2)
p.add_argument('--timeout', type=float, default=3600)
a = p.parse_args()
if a.interval <= 0 or a.timeout <= 0:
    p.error('interval and timeout must be positive')
dest = Path(a.output)
dest.parent.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
with dest.open('w', encoding='utf-8', buffering=1) as out:
    while time.monotonic() - started < a.timeout:
        sample = {'timestamp': time.time(), 'container': a.container, 'device': a.device}
        for key, command in (
            ('gpu', ['xpu-smi', 'stats', '-d', a.device, '-j']),
            ('stats', ['docker', 'stats', '--no-stream', '--format', '{{json .}}', a.container]),
        ):
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=15)
                sample[key] = json.loads(result.stdout)
                if result.returncode:
                    sample[key]['collector_error'] = result.stderr.strip() or f'exit {result.returncode}'
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                sample[key] = {'collector_error': str(exc)}
        sample['sample_end_timestamp'] = time.time()
        out.write(json.dumps(sample) + '\n')
        time.sleep(a.interval)
