# Stewards — Setup Guide

## 1. Pull the latest version

```bash
git pull
```

This will also update the `.gitignore`, which excludes the large data folders and local config files.

---

## 2. Download and place the data folder

Download the `stewards_files` folder (shared separately) and place it inside the `./backend` directory:

```
stewards/
└── backend/
    └── stewards_files/
        └── boston/
            ├── tiles/
            ├── masks_tile2net_polygons/
            ├── masks_confidence/
            ├── masks_groundtruth_polygons/
            └── stewards_scripts/
                ├── helper_scripts/
                └── output/
```

---

## 3. Create your `.env` file

Create a `.env` file at the project root (`stewards/.env`) and fill in the paths to match your local machine. You also need a Google Maps API key for the interface.

```dotenv
# ── Data directories ──
TILES_DIR=C:/Users/your_user/.../stewards/backend/stewards_files/boston/tiles
T2N_DIR=C:/Users/your_user/.../stewards/backend/stewards_files/boston/masks_tile2net_polygons
CONF_DIR=C:/Users/your_user/.../stewards/backend/stewards_files/boston/masks_confidence
GT_DIR=C:/Users/your_user/.../stewards/backend/stewards_files/boston/masks_groundtruth_polygons

# ── Global polygon / network files ──
ORIGINAL_POLYGONS=C:/Users/your_user/.../stewards/public/polygons.geojson
ORIGINAL_NETWORK=C:/Users/your_user/.../stewards/public/network.geojson

# ── Trained model output ──
TRANED_MODEL_OUTPUT=C:/Users/your_user/.../stewards/backend/stewards_files/boston/outputs

# ── Helper scripts path ──
HELPERS_PATH=C:/Users/your_user/.../stewards/backend/stewards_files/boston/stewards_scripts/helper_scripts

# ── Stewards scripts ──
SCRIPT_PATH=C:/Users/your_user/.../stewards/backend/stewards_files/boston/stewards_scripts

# ── Google Maps API key ──
VITE_GOOGLE_MAPS_KEY=your_key_here
```

> Use forward slashes (`/`) in all paths, even on Windows — Python handles them correctly on all platforms.

---

## 4. Install Python dependencies

From the `./backend` folder:

```bash
pip install -r requirements.txt
```

---

## 5. Run the project

You need **3 terminals** running simultaneously.

**Terminal 1 — Frontend** (from project root):
```bash
npm run dev
```

**Terminal 2 — Training server** (from `./backend`):
```bash
uvicorn server:app --reload --port 8001
```

**Terminal 3 — Map tiles server** (from project root):
```bash
python -m http.server 8002 --directory ./backend/stewards_files/map_tiles
```

Once all three are running, open [http://localhost:5173](http://localhost:5173) in your browser.