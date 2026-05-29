#!/usr/bin/env python3
"""
Batch-drive ComfyUI (FLUX GGUF + PuLID) over many scene prompts.

Face identity is fixed in workflow_api.json (LoadImage node "7" -> my_face.jpg);
only the positive prompt text + seed change per job. A cool-down delay between
jobs lets the GPU drop temperature.

Node IDs below already match the provided workflow_api.json:
  "9"  = positive CLIPTextEncode  (scene prompt goes here)
  "13" = KSampler                 (seed goes here)

Setup:
  1. ComfyUI running as API server (systemctl service) on SERVER below.
  2. Put your face crop in ComfyUI/input/  named  my_face.jpg
     (or change the name in workflow_api.json node "7").
  3. Put workflow_api.json next to this script.
  4. Put one scene prompt per line in scenes.txt
  5. python comfy_batch.py
"""

import json
import time
import random
import urllib.request
import urllib.error
from pathlib import Path

# ---------- CONFIG ----------
SERVER       = "http://127.0.0.1:8188"
WORKFLOW     = "workflow_api.json"
SCENES_FILE  = "scenes.txt"

POSITIVE_NODE_ID = "9"    # CLIPTextEncode (positive) in workflow_api.json
KSAMPLER_NODE_ID = "13"   # KSampler in workflow_api.json

DELAY_SECONDS    = 90     # cool-down between jobs (60-120s)
RANDOM_SEED      = True    # new seed per scene; False = fixed for A/B testing
FIXED_SEED       = 42
POLL_SECONDS     = 3       # how often to check if a job finished
JOB_TIMEOUT      = 900     # give up on a single image after this many seconds
# ----------------------------


def load_workflow():
    return json.loads(Path(WORKFLOW).read_text())


def load_scenes():
    lines = Path(SCENES_FILE).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"{SERVER}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["prompt_id"]


def is_done(prompt_id):
    try:
        with urllib.request.urlopen(f"{SERVER}/history/{prompt_id}") as resp:
            hist = json.loads(resp.read())
        return prompt_id in hist and bool(hist[prompt_id].get("outputs"))
    except urllib.error.URLError:
        return False


def wait_until_done(prompt_id):
    start = time.time()
    while time.time() - start < JOB_TIMEOUT:
        if is_done(prompt_id):
            return True
        time.sleep(POLL_SECONDS)
    print(f"  ! timeout waiting for {prompt_id}, moving on")
    return False


def main():
    wf_template = load_workflow()
    scenes = load_scenes()
    total = len(scenes)
    print(f"Loaded {total} scenes. Delay {DELAY_SECONDS}s between jobs.\n")

    for i, scene in enumerate(scenes, 1):
        wf = json.loads(json.dumps(wf_template))  # deep copy

        wf[POSITIVE_NODE_ID]["inputs"]["text"] = scene

        seed = random.randint(0, 2**32 - 1) if RANDOM_SEED else FIXED_SEED
        wf[KSAMPLER_NODE_ID]["inputs"]["seed"] = seed

        print(f"[{i}/{total}] seed={seed}  {scene[:70]}")

        try:
            pid = queue_prompt(wf)
        except urllib.error.URLError as e:
            print(f"  ! could not reach ComfyUI: {e}. Is the service up?")
            time.sleep(DELAY_SECONDS)
            continue

        wait_until_done(pid)
        print(f"  done. cooling down {DELAY_SECONDS}s...")

        if i < total:
            time.sleep(DELAY_SECONDS)

    print("\nAll scenes processed.")


if __name__ == "__main__":
    main()
