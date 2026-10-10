"""Interpret sampled xpu-smi device VRAM; never substitute host RAM."""
import json


def summarize_vram(path, start, end):
    values, timestamps, devices = [], [], set()
    invalid = 0
    if path:
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                sample = json.loads(line)
                timestamp = sample['timestamp']
                if not start <= timestamp <= end:
                    continue
                gpu = sample.get('gpu', {})
                if gpu.get('collector_error') or gpu.get('error'):
                    continue
                # Schema verified on the B70 xpu-smi installed on this host.
                tiles = gpu.get('memory', {}).get('used_mib', {})
                if not tiles:
                    continue
                mib = [float(tile['current']) for tile in tiles.values()]
                if any(v < 0 for v in mib):
                    continue
                values.append(sum(mib) / 1024)
                timestamps.append(timestamp)
                devices.add(gpu.get('pci_bdf', sample.get('device', 'unknown')))
            except (ValueError, TypeError, KeyError):
                invalid += 1
    if len(devices) > 1:
        raise ValueError('Metrics contain multiple GPU devices; use one B70 metrics file')
    return {
        'gpu_vram_peak_gib_sampled': max(values) if values else None,
        'gpu_device': next(iter(devices), None),
        'gpu_vram_sample_count': len(values),
        'first_vram_sample_timestamp': min(timestamps) if timestamps else None,
        'last_vram_sample_timestamp': max(timestamps) if timestamps else None,
        'vram_start_gap_seconds': min(timestamps) - start if timestamps else None,
        'vram_end_gap_seconds': end - max(timestamps) if timestamps else None,
        'invalid_metric_lines': invalid,
        'memory_scope': 'xpu-smi device-wide memory.used_mib current summed over tiles; sampled observed peak, includes other GPU processes and driver allocations; not torch allocated or host RAM',
    }
