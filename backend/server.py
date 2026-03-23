import json
import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Load .env from project root (two levels up from backend/) ─────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths from .env ────────────────────────────────────────────────────────────

SCRIPT_PATH  = Path(os.environ["SCRIPT_PATH"]) / "train_from_suggestions.py"
TILES_DIR    = Path(os.environ["TILES_DIR"])
T2N_DIR      = Path(os.environ["T2N_DIR"])
CONF_DIR     = Path(os.environ["CONF_DIR"])
MODEL_OUTPUT = Path(os.environ["TRAINED_MODEL_OUTPUT"]) / "suggestion_model.pt"

# ── In-memory job store ────────────────────────────────────────────────────────
# Each job: { status, message, epoch, total_epochs }

jobs: dict[str, dict] = {}


def _run_training(job_id: str, geojson_data: dict) -> None:
    try:
        proc = subprocess.Popen(
            [
                sys.executable, "-u",
                str(SCRIPT_PATH),
                "--tiles_dir",    str(TILES_DIR),
                "--t2n_dir",      str(T2N_DIR),
                "--conf_dir",     str(CONF_DIR),
                "--model_output", str(MODEL_OUTPUT),
                "--epochs",       "200",
                "--head",         "fix",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
        )

        # Send GeoJSON via stdin, then close so the script sees EOF
        proc.stdin.write(json.dumps(geojson_data))
        proc.stdin.close()

        # ── Parse epoch progress from stdout in real time ──
        stdout_lines = []
        epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)")

        for line in proc.stdout:
            stdout_lines.append(line)
            m = epoch_re.search(line)
            if m:
                epoch = int(m.group(1))
                total = int(m.group(2))
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


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/api/train")
async def start_training(body: dict) -> dict:
    if body.get("type") != "FeatureCollection" or not isinstance(body.get("features"), list):
        raise HTTPException(status_code=400, detail="Expected a GeoJSON FeatureCollection.")
    if not body["features"]:
        raise HTTPException(status_code=400, detail="No features in FeatureCollection.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "message": "", "epoch": 0, "total_epochs": 200}

    threading.Thread(target=_run_training, args=(job_id, body), daemon=True).start()

    return {"job_id": job_id}


@app.get("/api/train/status/{job_id}")
async def get_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job

APPLY_MODEL_SCRIPT = Path(os.environ["SCRIPT_PATH"]) / "apply_model.py"
ORIGINAL_POLYGONS  = Path(os.environ["ORIGINAL_POLYGONS"])
ORIGINAL_NETWORK   = Path(os.environ["ORIGINAL_NETWORK"])
OUTPUT_POLYGONS    = Path(os.environ["OUTPUT_POLYGONS"])
OUTPUT_NETWORK     = Path(os.environ["OUTPUT_NETWORK"])

inference_jobs: dict[str, dict] = {}

def _run_inference(job_id: str, tile_ids: list[str]) -> None:
    print(f"\n[INFERENCE START — job {job_id}]")
    print(f"  Tile count : {len(tile_ids)}")
    print(f"  Tile IDs   : {', '.join(tile_ids[:6])}{'…' if len(tile_ids) > 6 else ''}")
    try:
        proc = subprocess.Popen(
            [
                sys.executable, "-u", str(APPLY_MODEL_SCRIPT),
                "--tile_ids",          *tile_ids,
                "--model_path",        str(MODEL_OUTPUT),
                "--tiles_dir",         str(TILES_DIR),
                "--t2n_dir",           str(T2N_DIR),
                "--conf_dir",          str(CONF_DIR),
                "--original_polygons", str(ORIGINAL_POLYGONS),
                "--original_network",  str(ORIGINAL_NETWORK),
                "--output_polygons",   str(OUTPUT_POLYGONS),
                "--output_network",    str(OUTPUT_NETWORK),
                "--head", "fix",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
        )
 
        # Stream stdout line-by-line terminal + live job status
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  [inference] {line}")
                inference_jobs[job_id]["message"] = line
 
        proc.wait()
        stderr = proc.stderr.read()
 
        if proc.returncode == 0:
            print(f"[INFERENCE DONE — job {job_id}]\n")
            inference_jobs[job_id] = {"status": "done", "message": ""}
        else:
            stderr_tail = stderr[-800:] if stderr else ""
            print(f"\n[INFERENCE ERROR — job {job_id}]\nSTDERR:\n{stderr_tail}\n")
            inference_jobs[job_id] = {"status": "error", "message": stderr_tail or "Unknown error."}
 
    except Exception as exc:
        print(f"\n[INFERENCE EXCEPTION — job {job_id}]\n{exc}\n")
        inference_jobs[job_id] = {"status": "error", "message": str(exc)}

@app.post("/api/apply-model")
async def start_inference(body: dict) -> dict:
    tile_ids = body.get("tileIds", [])
    if not tile_ids:
        raise HTTPException(status_code=400, detail="No tile IDs provided.")
    job_id = str(uuid.uuid4())
    inference_jobs[job_id] = {"status": "running", "message": ""}
    threading.Thread(target=_run_inference, args=(job_id, tile_ids), daemon=True).start()
    return {"job_id": job_id}

@app.get("/api/apply-model/status/{job_id}")
async def get_inference_status(job_id: str) -> dict:
    job = inference_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job