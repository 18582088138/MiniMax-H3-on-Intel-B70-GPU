"""Submit a reproducible MiniMax H3 T2V workflow to an existing ComfyUI service."""
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error
import urllib.parse
import time
import re
from performance_metrics import summarize_vram

PROJECT_DIR = Path(__file__).resolve().parent


def project_path(value):
    """Resolve relative client paths against the project, independent of cwd."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def lora_filename_tag(lora):
    if not lora:
        return 'NoLoRA'
    stem = Path(lora.replace('\\', '/')).stem
    return 'LoRA-' + (re.sub(r'[^A-Za-z0-9._-]+', '_', stem).strip('._') or 'unnamed')


def workflow(width=256, height=160, frames=124, steps=20, seed=42, prompt_text=None):
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    return {
        "6": node("UNETLoader", unet_name="minimax_h3_fl2va_pruned_int8_convrot.safetensors", weight_dtype="default"),
        "13": node("CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default"),
        "11": node("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors"),
        "24": node("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors"),
        "104": node("MiniMaxH3ImageToVideo", clip=["13", 0], vae=["11", 0], prompt=prompt_text or "A small red sailboat moves slowly across a calm blue lake in daylight. A steady wide camera shot. Audio: gentle water ripples and soft wind. No text or subtitles.", width=width, height=height, length=frames),
        "15": node("RandomNoise", noise_seed=seed),
        "16": node("BasicGuider", model=["6", 0], conditioning=["104", 0]),
        "17": node("KSamplerSelect", sampler_name="res_multistep"),
        "9": node("BasicScheduler", model=["6", 0], scheduler="simple", steps=steps, denoise=1.0),
        "14": node("SamplerCustomAdvanced", noise=["15", 0], guider=["16", 0], sampler=["17", 0], sigmas=["9", 0], latent_image=["104", 1]),
        "10": node("VAEDecode", samples=["14", 0], vae=["11", 0]),
        "23": node("VAEDecodeAudio", samples=["14", 0], vae=["24", 0]),
        "91": node("CreateVideo", images=["10", 0], audio=["23", 0], fps=24, bit_depth=8),
        "92": node("SaveVideo", video=["91", 0], filename_prefix=f"video/B70_MiniMaxH3_{width}x{height}_{frames}frames_{steps}steps", format="mp4", codec="auto"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", help="Description of video and native audio")
    parser.add_argument("--prompt-file", help="UTF-8 file containing a detailed video/audio prompt")
    parser.add_argument("--tiled-vae", action="store_true", help="Decode video VAE in spatial and temporal tiles")
    parser.add_argument("--lora", help="ComfyUI LoRA filename relative to models/loras")
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--metrics-file", help="JSONL device VRAM collected by collect_metrics.py on the B70 host")
    parser.add_argument("--history", metavar="PROMPT_ID")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--output", default="workflows/workflow_api.json")
    parser.add_argument("--wait", action="store_true", help="Wait for completion and save history and output files locally")
    parser.add_argument("--timeout", type=float, default=1800, help="Maximum wait in seconds")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Result root; videos and history go into separate subdirectories")
    args = parser.parse_args()
    if args.wait and not args.submit:
        parser.error("--wait requires --submit")
    if args.prompt and args.prompt_file:
        parser.error("use either --prompt or --prompt-file")
    if args.width < 32 or args.height < 32 or args.width % 32 or args.height % 32:
        parser.error("width and height must be positive multiples of 32")
    if args.frames < 5 or (args.frames - 5) % 17:
        parser.error("frames must follow the MiniMax 17k+5 grid, such as 22 or 124")
    if args.steps < 1 or args.timeout <= 0:
        parser.error("steps and timeout must be positive")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if args.history:
        print(opener.open(args.url + "/history/" + args.history, timeout=30).read().decode())
        return
    prompt_text = project_path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    prompt = workflow(args.width, args.height, args.frames, args.steps, args.seed, prompt_text)
    if args.lora:
        prompt['30'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['6', 0], 'lora_name': args.lora, 'strength_model': args.lora_strength}}
        prompt['16']['inputs']['model'] = ['30', 0]
        prompt['9']['inputs']['model'] = ['30', 0]
    lora_tag = lora_filename_tag(args.lora)
    prompt['92']['inputs']['filename_prefix'] += '_' + lora_tag
    print(f"Requested {args.width}x{args.height}, {args.frames} frames at 24 fps "
          f"({args.frames / 24:.3f} seconds), {args.steps} sampling steps, seed {args.seed}.", flush=True)
    if args.tiled_vae:
        prompt["10"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["14", 0], "vae": ["11", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}}
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")
    if args.submit:
        request = urllib.request.Request(args.url + "/prompt", data=json.dumps({"prompt": prompt, "client_id": "b70-minimax-verification"}).encode(), headers={"Content-Type": "application/json"})
        try:
            result = json.load(opener.open(request, timeout=30))
            print(json.dumps(result), flush=True)
        except urllib.error.HTTPError as exc:
            print(exc.read().decode())
            raise SystemExit(1)
        except (OSError, urllib.error.URLError) as exc:
            print(f"Submission did not receive a response: {exc}. Check service/tunnel and queue before retrying; acceptance is unknown.", flush=True)
            raise SystemExit(3)
        if args.wait:
            prompt_id = result["prompt_id"]
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                try:
                    history = json.load(opener.open(args.url + "/history/" + prompt_id, timeout=30))
                except (OSError, urllib.error.URLError) as exc:
                    print(f"Cannot read history for {prompt_id}: {exc}. Check SSH tunnel and ComfyUI/container logs; generation completion is unknown.", flush=True)
                    raise SystemExit(3)
                if prompt_id in history:
                    record = history[prompt_id]
                    artifact_dir = project_path(args.artifacts_dir)
                    history_dir = artifact_dir / "history"
                    video_dir = artifact_dir / "videos"
                    history_dir.mkdir(parents=True, exist_ok=True)
                    video_dir.mkdir(parents=True, exist_ok=True)
                    history_path = history_dir / f"history_{prompt_id}.json"
                    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
                    if record["status"]["status_str"] != "success":
                        print(f"Generation failed; see {history_path}", flush=True)
                        raise SystemExit(1)
                    stamps = {kind: data['timestamp'] for kind, data in record['status'].get('messages', []) if kind in ('execution_start', 'execution_success') and 'timestamp' in data}
                    elapsed = (stamps['execution_success'] - stamps['execution_start']) / 1000 if all(k in stamps for k in ('execution_start', 'execution_success')) else None
                    metrics_path = project_path(args.metrics_file) if args.metrics_file else None
                    report = summarize_vram(metrics_path, stamps['execution_start'] / 1000, stamps['execution_success'] / 1000) if elapsed is not None else summarize_vram(None, 0, 0)
                    vram_peak = report['gpu_vram_peak_gib_sampled']
                    report.update(prompt_id=prompt_id, execution_seconds=elapsed, lora=args.lora)
                    report_dir = artifact_dir / 'reports'
                    report_dir.mkdir(parents=True, exist_ok=True)
                    (report_dir / f'performance_{prompt_id}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                    for output in record["outputs"].values():
                        for item in output.get("images", []):
                            if item.get("type") == "output":
                                query = urllib.parse.urlencode({k: item[k] for k in ("filename", "subfolder", "type")})
                                data = opener.open(args.url + "/view?" + query, timeout=60).read()
                                original = Path(item['filename'])
                                elapsed_tag = f'{elapsed:.2f}s' if elapsed is not None else 'NA'
                                vram_tag = f'{vram_peak:.2f}GiB' if vram_peak is not None else 'NA'
                                # SaveVideo may append its numeric counter after the tag.
                                stem = original.stem if '_' + lora_tag in original.stem else original.stem + '_' + lora_tag
                                target = video_dir / f'{stem}_time{elapsed_tag}_VRAMpeak{vram_tag}{original.suffix}'
                                target.write_bytes(data)
                                print(f"Saved {target} ({len(data)} bytes)", flush=True)
                    print(f"Generation succeeded; saved {history_path}", flush=True)
                    break
                time.sleep(5)
            else:
                print(f"Wait timed out; prompt {prompt_id} may still be running.", flush=True)
                raise SystemExit(2)
    else:
        print(f"Saved workflow to {output_path}; use --submit to execute.")


if __name__ == "__main__":
    main()
