import numpy as np
import scipy.ndimage
from config import MAP_SIZE, ROBOT_RADIUS, EXTRA_MARGIN, PENALTY_BAND, PENALTY_GAIN

def infer_resolution_from_map(occ_map, map_size_m=MAP_SIZE):
    H, W = occ_map.shape
    assert H == W, f"Expected square map, got {H}x{W}"
    return map_size_m / float(H), H

def downsample_max(binary_map, factor):
    H, W = binary_map.shape
    pad_h = (-H) % factor; pad_w = (-W) % factor
    if pad_h or pad_w:
        binary_map = np.pad(binary_map, ((0, pad_h),(0, pad_w)), mode='edge')
        H, W = binary_map.shape
    return binary_map.reshape(H//factor, factor, W//factor, factor).max(axis=(1,3)).astype(binary_map.dtype)

def world_to_grid_res(x, y, res):
    gx = int((x + MAP_SIZE / 2.0) / res)
    gy = int((y + MAP_SIZE / 2.0) / res)
    return gx, gy

def grid_to_world_res(gx, gy, res):
    x = gx * res - MAP_SIZE/2.0 + res/2.0
    y = gy * res - MAP_SIZE/2.0 + res/2.0
    return x, y

def build_planning_layers(binary_occ, resolution):
    free = (binary_occ == 0)
    edt = scipy.ndimage.distance_transform_edt(free).astype(np.float32) * resolution
    hard_clear = ROBOT_RADIUS + EXTRA_MARGIN
    inflated_occ = (edt < hard_clear).astype(np.uint8)
    band = np.clip(PENALTY_BAND - edt, 0.0, PENALTY_BAND)
    penalty_map = (band / max(PENALTY_BAND, 1e-6)) * PENALTY_GAIN
    return inflated_occ, penalty_map.astype(np.float32), edt

def force_goal_free(combined_occ, goal_g, radius_cells=2):
    H, W = combined_occ.shape
    gx, gy = goal_g
    if not (0 <= gx < H and 0 <= gy < W): return
    rr = int(max(1, radius_cells))
    for i in range(max(0,gx-rr), min(H-1, gx+rr)+1):
        for j in range(max(0,gy-rr), min(W-1, gy+rr)+1):
            if (i-gx)*(i-gx) + (j-gy)*(j-gy) <= rr*rr:
                combined_occ[i, j] = 0