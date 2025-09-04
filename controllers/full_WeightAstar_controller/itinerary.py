import csv, os, re
from config import COL_ID, COL_ITIN
from named_points import NAMED_POINTS

def normalize_name(s: str) -> str:
    s = re.sub(r'^\s*\[.*?\]\s*', '', s)
    s = s.strip().lower()
    s = re.sub(r"[—\-]+", "-", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def parse_itinerary_text(text: str):
    names = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower() in ("entrance", "exit") or line.lower().startswith("includes"):
            continue
        parts = [p.strip() for p in re.split(r"—|-", line) if p.strip()]
        if len(parts) >= 2:
            names.append(parts[1])
    return names

def load_names_from_csv(csv_path: str, unique_id: str):
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}")
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for row in rows:
        if len(row) <= max(COL_ID, COL_ITIN):
            continue
        if row[COL_ID].strip() == unique_id:
            return parse_itinerary_text(row[COL_ITIN])
    print(f"[WARN] Unique id not found in CSV: {unique_id}")
    return []

def names_to_waypoints(names):
    wps = []
    for name in names:
        key = normalize_name(name)
        meta = NAMED_POINTS.get(key)
        if not meta:
            print(f"[WARN] Missing named point for '{name}' (key='{key}'), skipping.")
            continue
        (x, y) = meta["xy"]
        wps.append((x, y, name))
    return wps