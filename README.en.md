[Українська](README.md) | English

# Generating 1000 Images with ComfyUI + FLUX

A reproducible writeup of the experiment: running ComfyUI locally as an API service with the FLUX model (GGUF Q4) on a single RTX 3060 (12 GB), and batch-generating 1000 photorealistic scenes via a Python script.

## Hardware and system requirements

- GPU: NVIDIA RTX 3060, 12 GB VRAM (compute capability 8.6)
- RAM: at least 32 GB (for weight offloading and pinned memory)
- OS: Ubuntu (Linux x86_64)
- NVIDIA drivers with CUDA 12.4 support
- Python 3.12, the `uv` package manager
- Free disk space: ~20 GB for models

## 1. Installing ComfyUI in a venv (uv)

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
uv venv --python 3.12
source .venv/bin/activate

# PyTorch built for CUDA 12.4
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt

# GGUF loader node (required for Q4 FLUX)
cd custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF.git
cd ComfyUI-GGUF && uv pip install -r requirements.txt
cd ../..
```

### Common issue: torchaudio

If startup fails with `OSError: libcudart.so.13: cannot open shared object file`, the `torchaudio` version does not match the `torch` version. Fix:

```bash
uv pip install torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

ComfyUI does not need `torchaudio` for image generation — you can simply remove it if needed (`uv pip uninstall torchaudio`).

## 2. Downloading the models

```bash
uv pip install huggingface_hub

# FLUX Q4 model -> models/unet
hf download city96/FLUX.1-dev-gguf flux1-dev-Q4_K_S.gguf --local-dir models/unet

# T5 Q4 text encoder -> models/clip
hf download city96/t5-v1_1-xxl-encoder-gguf t5-v1_1-xxl-encoder-Q4_K_S.gguf --local-dir models/clip

# clip_l -> models/clip
hf download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/clip

# VAE -> models/vae
hf download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir models/vae
```

> FLUX.1-dev is a gated model. First accept the license on its Hugging Face page and run `hf auth login`. An ungated alternative is the `schnell` model.

Verify file placement:

```bash
ls models/unet/   # flux1-dev-Q4_K_S.gguf
ls models/clip/   # t5-v1_1-xxl-encoder-Q4_K_S.gguf, clip_l.safetensors
ls models/vae/    # ae.safetensors
```

## 3. Running ComfyUI as a systemd service

File `/etc/systemd/system/comfyui.service` (substitute your user and paths):

```ini
[Unit]
Description=ComfyUI API server
After=network.target

[Service]
Type=simple
User=automation
Group=automation
WorkingDirectory=/home/automation/ComfyUI
ExecStart=/home/automation/ComfyUI/.venv/bin/python main.py --listen 127.0.0.1 --port 8188 --cache-lru 1
Restart=on-failure
RestartSec=5
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[Install]
WantedBy=multi-user.target
```

Install and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now comfyui
systemctl status comfyui
journalctl -u comfyui -f          # live logs
curl http://127.0.0.1:8188/system_stats
```

### Note on VRAM and model reloading

- The `--lowvram` flag offloads model weights to RAM between runs, which causes FLUX to **reload every time**. On 12 GB the Q4 model (~6.6 GB) fits without it.
- To keep the model resident between requests, use normal VRAM mode + `--cache-lru 1` (as in the unit above).
- If you get an OOM at the VAE decode step without `--lowvram`, add `--reserve-vram 0.5`.
- ComfyUI claims the GPU as soon as it starts. Before running vLLM/Ollama on the same card, stop the service: `sudo systemctl stop comfyui`.

## 4. Workflow file (API format)

Over the API, ComfyUI accepts a complete workflow graph as JSON — not just a model name and a prompt. The `workflow_api.json` graph for FLUX GGUF (text-to-image, no PuLID):

```json
{
  "1": { "class_type": "UnetLoaderGGUF",
         "inputs": { "unet_name": "flux1-dev-Q4_K_S.gguf" } },
  "2": { "class_type": "DualCLIPLoaderGGUF",
         "inputs": { "clip_name1": "t5-v1_1-xxl-encoder-Q4_K_S.gguf",
                     "clip_name2": "clip_l.safetensors", "type": "flux" } },
  "3": { "class_type": "VAELoader",
         "inputs": { "vae_name": "ae.safetensors" } },
  "9": { "class_type": "CLIPTextEncode",
         "inputs": { "clip": ["2", 0],
                     "text": "a man standing in a city street, photorealistic" } },
  "11": { "class_type": "FluxGuidance",
          "inputs": { "conditioning": ["9", 0], "guidance": 3.5 } },
  "12": { "class_type": "EmptyLatentImage",
          "inputs": { "width": 896, "height": 896, "batch_size": 1 } },
  "13": { "class_type": "KSampler",
          "inputs": { "model": ["1", 0], "positive": ["11", 0], "negative": ["9", 0],
                      "latent_image": ["12", 0], "seed": 42, "steps": 20,
                      "cfg": 1.0, "sampler_name": "euler",
                      "scheduler": "simple", "denoise": 1.0 } },
  "14": { "class_type": "VAEDecode",
          "inputs": { "samples": ["13", 0], "vae": ["3", 0] } },
  "15": { "class_type": "SaveImage",
          "inputs": { "images": ["14", 0], "filename_prefix": "scene" } }
}
```

Key points:

- For FLUX, `cfg` in KSampler = **1.0**; prompt strength is controlled by the separate `FluxGuidance` node (3.5). Raising `cfg` degrades the result.
- The negative prompt is wired to node `9` as a placeholder (FLUX ignores it at `cfg=1.0`).
- Plain `VAEDecode` is used. If you hit an OOM at higher resolutions, switch to `VAEDecodeTiled` with the `tile_size`, `overlap`, `temporal_size`, `temporal_overlap` parameters.
- The GGUF node names (`UnetLoaderGGUF`, `DualCLIPLoaderGGUF`) depend on the extension version. Check the actual names:

```bash
curl -s http://127.0.0.1:8188/object_info | grep -io '"UnetLoaderGGUF"\|"DualCLIPLoaderGGUF"'
```

### Resolution

- Base: **896×896** (safe for 12 GB).
- FLUX is trained at ~1 MP. For 16:9, it is better to generate close to 1 MP (e.g. **1344×768**) and upscale in a separate pass if needed, rather than going straight to 1920×1080 (which risks composition artifacts and OOM).

## 5. Scenes file (scenes.txt)

One prompt per line; lines starting with `#` are skipped. The emphasis is on the action and environment; the man is a mid-ground figure, not a posed model. Example line:

```
a man herding sheep across an alpine meadow high in the Ukrainian Carpathians; wildflower grass, scattered flock, distant snow-capped peaks dominating the horizon; the man as a mid-ground figure about a third of the frame height, body and action clearly visible but not dominant; hazy warm summer afternoon; wide 16:9 composition, deep depth of field, environment and action as joint focal points, photorealistic, highly detailed
```

Prompt structure: `action + location; environment detail; figure scale; lighting; quality tags`.

## 6. Batch script (comfy_batch.py)

The script sequentially reads lines from `scenes.txt`, substitutes the text into node `9` and the seed into node `13`, posts the graph to `/prompt`, waits for completion via `/history`, then pauses to let the GPU cool down.

Key parameters (at the top of the file):

| Parameter | Value | Purpose |
|---|---|---|
| `SERVER` | `http://127.0.0.1:8188` | ComfyUI API address |
| `WORKFLOW` | `workflow_api.json` | graph in API format |
| `SCENES_FILE` | `scenes.txt` | prompt list |
| `POSITIVE_NODE_ID` | `"9"` | positive prompt node |
| `KSAMPLER_NODE_ID` | `"13"` | KSampler node (seed) |
| `DELAY_SECONDS` | `90` | pause between jobs (cool-down) |
| `RANDOM_SEED` | `True` | new seed per scene |
| `JOB_TIMEOUT` | `900` | timeout for one image (sec) |

### Client file locations

Keep the client files (`workflow_api.json`, `comfy_batch.py`, `scenes.txt`) in a separate folder, **not** in the ComfyUI directory:

```
~/comfy_ui_scripts/
├── workflow_api.json
├── comfy_batch.py
├── scenes.txt
└── batch.log
```

The script reads `workflow_api.json` and `scenes.txt` from the current directory, so run it from there. The generated images are saved by the ComfyUI server itself into its `output/` folder — independent of where the client runs.

```bash
mkdir -p ~/comfy_ui_scripts
mv workflow_api.json comfy_batch.py scenes.txt ~/comfy_ui_scripts/
```

Launch (survives SSH session disconnect):

```bash
cd ~/comfy_ui_scripts
nohup python comfy_batch.py > batch.log 2>&1 &
tail -f batch.log
```

> `nohup` survives an SSH drop but **not** a server reboot or an OOM-kill. For full resilience, run it under `tmux`/`screen` or as a separate systemd unit.

## 7. Results

Images are saved in ComfyUI's standard output folder:

```bash
ls /home/automation/ComfyUI/output/   # scene_00001_.png, scene_00002_.png, ...
```

## Approximate timing

At ~25 s per image + 90 s pause, one scene takes ≈2 min; 1000 scenes is roughly **30+ hours** of total runtime. Plan for two overnight sessions or reduce `DELAY_SECONDS`.

## Final file manifest

```
~/comfy_ui_scripts/          # client files (separate folder)
├── workflow_api.json        # workflow graph (API format)
├── comfy_batch.py           # batch driver
├── scenes.txt               # 1000 prompts (1 per line)
└── batch.log                # run log (nohup)

~/ComfyUI/                   # the server itself
└── output/                  # generated images (scene_*.png)
```

## Planned improvements

- Raise resolution to Full HD via an extra upscale pass (generate 1344×768 → upscale to 1920×1080).
- Move to a larger model (Q5_K_M ≈ 8 GB) if VRAM headroom allows.
- If a specific face needs to be locked in, return to PuLID-Flux + a face photo.
