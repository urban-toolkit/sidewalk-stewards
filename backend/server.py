import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE = Path(r"C:\Users\leo_v\Documents\leovsferreira\code\tile2net\dorchester_exports")

SCRIPT_PATH    = BASE / "stewards_scripts" / "train_from_suggestions.py"
TILES_DIR      = BASE / "tiles"
T2N_DIR        = BASE / "masks_tile2net_polygons"
CONF_DIR       = BASE / "masks_confidence"
OUTPUT_GEOJSON = BASE / "outputs" / "polygons.geojson"
MODEL_OUTPUT   = BASE / "outputs" / "suggestion_model.pt"

# ── In-memory job store ────────────────────────────────────────────────────────
# Each job: { status, message, epoch, total_epochs }

jobs: dict[str, dict] = {}


def _make_remapped_dir(source_dir: Path, suffix: str, tmp_parent: str) -> str:
    tmp = tempfile.mkdtemp(dir=tmp_parent)
    for src in source_dir.glob(f"*{suffix}.png"):
        tid = src.stem.replace(suffix, "")
        dst = Path(tmp) / f"{tid}.png"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return tmp


def _run_training(job_id: str, geojson_path: str) -> None:
    tmp_root = tempfile.mkdtemp(prefix="stewards_remap_")
    try:
        t2n_remapped  = _make_remapped_dir(T2N_DIR,  "_predictions", tmp_root)
        conf_remapped = _make_remapped_dir(CONF_DIR, "_prob_mask",   tmp_root)

        proc = subprocess.Popen(
        [
            sys.executable, "-u",
            str(SCRIPT_PATH),
            "--geojson",      geojson_path,
            "--tiles_dir",    str(TILES_DIR),
            "--t2n_dir",      t2n_remapped,
            "--conf_dir",     conf_remapped,
            "--output",       str(OUTPUT_GEOJSON),
            "--model_output", str(MODEL_OUTPUT),
            "--epochs",       "200",
            "--head",         "fix",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},  # ← also set env var
    )

        # ── Parse epoch progress from stdout in real time ──
        stdout_lines = []
        epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)")

        for line in proc.stdout:
            stdout_lines.append(line)
            m = epoch_re.search(line)
            if m:
                epoch      = int(m.group(1))
                total      = int(m.group(2))
                jobs[job_id]["epoch"]        = epoch
                jobs[job_id]["total_epochs"] = total

        proc.wait()
        stderr = proc.stderr.read()

        if proc.returncode == 0:
            jobs[job_id].update({
                "status":       "done",
                "message":      "Model trained successfully.",
                "epoch":        jobs[job_id].get("total_epochs", 200),
                "total_epochs": jobs[job_id].get("total_epochs", 200),
            })
        else:
            stdout_tail = "".join(stdout_lines[-60:])
            stderr_tail = stderr[-2000:] if stderr else ""
            full = f"STDOUT:\n{stdout_tail}\n\nSTDERR:\n{stderr_tail}"
            print(f"\n[TRAINING ERROR — job {job_id}]\n{full}\n")
            jobs[job_id].update({"status": "error", "message": full})

    except Exception as exc:
        print(f"\n[TRAINING EXCEPTION — job {job_id}]\n{exc}\n")
        jobs[job_id].update({"status": "error", "message": str(exc)})
    finally:
        try:
            Path(geojson_path).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/api/train")
async def start_training(body: dict) -> dict:
    if body.get("type") != "FeatureCollection" or not isinstance(body.get("features"), list):
        raise HTTPException(status_code=400, detail="Expected a GeoJSON FeatureCollection.")
    if not body["features"]:
        raise HTTPException(status_code=400, detail="No features in FeatureCollection.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "message": "", "epoch": 0, "total_epochs": 200}

    tmp = tempfile.NamedTemporaryFile(
        suffix=".geojson", delete=False, mode="w", encoding="utf-8"
    )
    json.dump(body, tmp)
    tmp.close()

    threading.Thread(target=_run_training, args=(job_id, tmp.name), daemon=True).start()

    return {"job_id": job_id}


@app.get("/api/train/status/{job_id}")
async def get_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job