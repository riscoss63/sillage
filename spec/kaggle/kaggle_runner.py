"""Kaggle script kernel: run the full Sillage speculative-drafter GPU suite.

Self-diagnosing version: waits for the dataset mount (a fresh dataset can
still be processing when the kernel starts), checks the GPU's CUDA
capability (the preinstalled torch dropped sm_60, so a P100 cannot run it
-- the kernel then says exactly what to change and stops cleanly), runs
every config, renames the two C results, and leaves the JSONs plus this
log in /kaggle/working.
"""

import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sillage", "--no-deps"], check=True)

import torch  # noqa: E402

if not torch.cuda.is_available():
    sys.exit("Pas de GPU alloue : Settings -> Accelerator -> GPU T4 x2.")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {name} (sm_{cap[0]}{cap[1]}) | torch {torch.__version__}",
      flush=True)
if cap[0] < 7:
    print("\n" + "!" * 70)
    print(f"{name} (sm_{cap[0]}{cap[1]}) n'est PAS supporte par le torch "
          f"de l'image Kaggle (minimum sm_70).")
    print("Dans l'interface du kernel : Settings -> Accelerator -> "
          "GPU T4 x2, puis 'Save & Run All'. Rien d'autre a changer.")
    print("!" * 70, flush=True)
    sys.exit(0)

# ---- find the kit, wherever Kaggle mounts it (layout changed over time) --

def tree(root, depth=3):
    out = []
    base = root.rstrip("/").count("/")
    for r, dirs, files in os.walk(root):
        if r.count("/") - base >= depth:
            dirs[:] = []
            continue
        for d in dirs:
            out.append(os.path.join(r, d) + "/")
        for f in files[:5]:
            out.append(os.path.join(r, f))
        if len(out) > 40:
            break
    return out


kit_src = None
for attempt in range(12):                       # up to ~6 minutes
    hits = glob.glob("/kaggle/input/**/kaggle_kit", recursive=True)
    hits = [h for h in hits if os.path.isdir(h)]
    if hits:
        kit_src = hits[0]
        break
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall("/kaggle/working/_kitzip")
        cand = glob.glob("/kaggle/working/_kitzip/**/kaggle_kit",
                         recursive=True)
        if cand:
            kit_src = cand[0]
            break
    print(f"kit pas encore visible (essai {attempt + 1}/12) ; "
          f"arborescence de /kaggle/input :", flush=True)
    for line in tree("/kaggle/input"):
        print("   ", line, flush=True)
    time.sleep(30)
if kit_src is None:
    sys.exit("Le kit est introuvable dans /kaggle/input : verifier que le "
             "dataset sillage-spec-kit est bien attache au kernel.")
print("kit:", kit_src, flush=True)

kit = "/kaggle/working/kit"
shutil.copytree(kit_src, kit)
os.chdir(kit)

RUNS = [
    (["--config", "micro", "--target", "Qwen/Qwen3-1.7B"], "micro"),
    (["--config", "A", "--pld"], "A"),
    (["--config", "C", "--target", "Qwen/Qwen3-1.7B",
      "--beta", "40", "--lam", "0.85", "--thrq", "0.5"], "C_17b"),
    (["--config", "C", "--target", "Qwen/Qwen3-4B", "--calibrate"], "C_4b"),
    (["--config", "B", "--target", "Qwen/Qwen3-1.7B"], "B"),
    (["--config", "gpt2", "--pld"], "gpt2"),
]

for args, tag in RUNS:
    print("\n" + "=" * 70 + "\nRUN " + tag + ": " + " ".join(args),
          flush=True)
    rc = subprocess.run(
        [sys.executable, "bench_gpu.py", "--device", "cuda",
         "--dtype", "float16"] + args).returncode
    print("exit:", rc, flush=True)
    cfg = args[args.index("--config") + 1]
    produced = f"results_gpu_{cfg}.json"
    if os.path.exists(produced):
        shutil.move(produced, f"/kaggle/working/results_gpu_{tag}.json")

shutil.rmtree(kit, ignore_errors=True)   # keep the output artifact small
shutil.rmtree("/kaggle/working/_kitzip", ignore_errors=True)
print("\nDONE", flush=True)
