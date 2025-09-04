from controller import Robot, Keyboard
import numpy as np, math, matplotlib.pyplot as plt
import dynamic_SLAM
import time, csv
import os, sys, pathlib
sys.path.insert(0, os.path.dirname(__file__))
from matplotlib.colors import ListedColormap
from config import *
from ui import ControlServer
from itinerary import load_names_from_csv, names_to_waypoints
from named_points import NAMED_POINTS
from map_utils import (
    infer_resolution_from_map, downsample_max, build_planning_layers,
    force_goal_free
)
from map_utils import grid_to_world_res as to_world, world_to_grid_res as to_grid

from planner import weighted_a_star
from dwa import read_scan_layers, evaluate_dwa, vw_to_wheels, build_lidar_obstacles_layers
import scipy.ndimage  

# Episode/outcome thresholds
try:
    from config import CRASH_DIST, CRASH_STEPS, GOAL_TIMEOUT_SEC
except Exception:
    CRASH_DIST = 0.10
    CRASH_STEPS = 4
    GOAL_TIMEOUT_SEC = 3000


# stall tuning 
PROGRESS_RATE_MPS_MIN         = globals().get("PROGRESS_RATE_MPS_MIN", 0.08)  
PROGRESS_EPSILON              = globals().get("PROGRESS_EPSILON", 0.005)     
ANGLE_ALIGN_RAD               = globals().get("ANGLE_ALIGN_RAD", math.radians(25))  
STALL_MIN_DIST_M              = globals().get("STALL_MIN_DIST_M", 0.40)      
STALL_REPLAN_COOLDOWN_STEPS   = globals().get("STALL_REPLAN_COOLDOWN_STEPS", 80)
BLOCKED_REQUIRE_BOTH          = globals().get("BLOCKED_REQUIRE_BOTH", True)   

# Proactive dynamic replan gates
DYN_REPLAN_LOOKAHEAD_M    = globals().get("DYN_REPLAN_LOOKAHEAD_M", 3.0)
DYN_REPLAN_CORRIDOR_M     = globals().get("DYN_REPLAN_CORRIDOR_M", 0.55)
DYN_REPLAN_COOLDOWN_STEPS = globals().get("DYN_REPLAN_COOLDOWN_STEPS", 20)

# Online static learning 
LEARN_STATIC_ENABLE           = globals().get("LEARN_STATIC_ENABLE", True)
LEARN_STATIC_PERSIST_STEPS    = globals().get("LEARN_STATIC_PERSIST_STEPS", 45)  
LEARN_STATIC_MIN_DIAM_M       = globals().get("LEARN_STATIC_MIN_DIAM_M", 0.30)    
LEARN_STATIC_INFLATE_M        = globals().get("LEARN_STATIC_INFLATE_M", 0.20)     
LEARN_STATIC_DECAY_PER_STEP   = globals().get("LEARN_STATIC_DECAY_PER_STEP", 0.0) 
LEARN_STATIC_VEL_THRESH_MS    = globals().get("LEARN_STATIC_VEL_THRESH_MS", 0.10) 

def nearest_by_category(x, y, category: str):
    best = None; bestd = 1e18
    for k, meta in NAMED_POINTS.items():
        if meta.get("cat") != category: continue
        xx, yy = meta["xy"]
        d2 = (xx-x)*(xx-x) + (yy-y)*(yy-y)
        if d2 < bestd: bestd = d2; best = (xx, yy, k)
    return best

def pretty_name(key: str) -> str:
    return key.replace("_", " ").title()

def main():

    assert os.path.exists(OCC_FILE), f"Map file not found: {OCC_FILE}"
    occ_fine = np.load(OCC_FILE)
    RES_FINE, _ = infer_resolution_from_map(occ_fine, MAP_SIZE)
    RES_COARSE = RES_FINE * COARSE_FACTOR

    static_bin_fine = np.zeros_like(occ_fine, dtype=np.uint8)
    static_bin_fine[occ_fine > 0.6] = 1
    static_bin_fine[occ_fine < 0.4] = 0
    static_bin_coarse = downsample_max(static_bin_fine, COARSE_FACTOR)

    inflated_occ_static, penalty_map_static, _ = build_planning_layers(static_bin_coarse, RES_COARSE)


    static_free_coarse = 1 - inflated_occ_static.astype(np.uint8)
    STATIC_DT_M = scipy.ndimage.distance_transform_edt(static_free_coarse) * RES_COARSE

    STATIC_NEAR_TOL_M = globals().get("STATIC_NEAR_TOL_M", 0.25)

    #Devices
    robot = Robot()
    keyboard = Keyboard(); keyboard.enable(TIME_STEP)
    gps   = robot.getDevice("gps");            gps.enable(TIME_STEP)
    imu   = robot.getDevice("inertial unit");  imu.enable(TIME_STEP)
    lidar = robot.getDevice("lidar");          lidar.enable(TIME_STEP)
    left_motor  = robot.getDevice("wheel_left_joint")
    right_motor = robot.getDevice("wheel_right_joint")
    left_motor.setPosition(float('inf')); right_motor.setPosition(float('inf'))

    ui = ControlServer(port=8765)
    ui.set_state(status="waiting-id", target="", current_id="")

    path = []

    ALL_DEST = []
    KEY_BY_NAME = {}
    for k, meta in NAMED_POINTS.items():
        name = pretty_name(k)
        ALL_DEST.append({"name": name, "cat": meta.get("cat", "")})
        KEY_BY_NAME[name] = k
    ui.set_state(all_destinations=ALL_DEST)

    while robot.step(TIME_STEP) == -1:
        pass

    H = lidar.getHorizontalResolution()
    fov = lidar.getFov()
    max_range = min(lidar.getMaxRange(), 8.0)
    angles_h = np.linspace(-fov/2, fov/2, H)

    slam = dynamic_SLAM.DynamicSLAM(map_size_m=MAP_SIZE)
    slam.set_static_map_from_file(OCC_FILE)

    # Dynamic map for planning 
    dynamic_occ_coarse = np.zeros_like(inflated_occ_static, dtype=np.float32)

    DYN_DECAY_PER_STEP = globals().get("DYN_DECAY_PER_STEP", 0.06)
    DYN_MARK_VALUE     = globals().get("DYN_MARK_VALUE", 1.00)
    DYN_THRESH_BIN     = globals().get("DYN_THRESH_BIN", 0.50)

    DYN_HARD_INFLATE_M = globals().get("DYN_HARD_INFLATE_M", 0.55)
    DYN_SOFT_INFLATE_M = globals().get("DYN_SOFT_INFLATE_M", 1.20)
    DYN_COST_FALLOFF_M = globals().get("DYN_COST_FALLOFF_M", 0.90)
    DYN_COST_SCALE     = globals().get("DYN_COST_SCALE", 35.0)
    DYN_GOAL_RELIEF_M  = globals().get("DYN_GOAL_RELIEF_M", 0.60)

    # Learned static layers
    learned_static_mask = np.zeros_like(inflated_occ_static, dtype=np.uint8)  
    persist_counter     = np.zeros_like(inflated_occ_static, dtype=np.uint16) 
    
    def _disc_struct(radius_m, res):
        r = max(1, int(radius_m / res))
        se = np.zeros((2*r+1, 2*r+1), dtype=np.uint8)
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        se[xx*xx + yy*yy <= r*r] = 1
        return se

    SE_LEARN_CLOSE = _disc_struct(LEARN_STATIC_MIN_DIAM_M/2.0, RES_COARSE)   
    SE_LEARN_INFL  = _disc_struct(LEARN_STATIC_INFLATE_M, RES_COARSE)     

    current_unique_id = ""
    names = []; waypoints = []; cur_idx = 0; cur_target_label = None
    diverted_from_idx = None; RETURN_OFFER_IDX = None; AWAIT_RETURN_DECISION = False

    # Loiter behavior
    LOITER_RADIUS = 0.7
    LOITER_SPEED  = 0.20
    LOITER_ANG_RATE = 0.8
    REPLAN_INTERVAL_STEPS = 35

    LOITERING = False
    PENDING_GOAL_WORLD = None
    loiter_center = None
    loiter_phase = 0.0
    replan_tick = 0

    #DWA episode metrics
    class DwaEpisode:
        def __init__(self, uid: str, goal_label: str):
            self.uid = uid or "anon"; self.goal = goal_label or "Waypoint"
            self.t0 = time.time(); self.steps = 0; self.blocked_steps = 0
            self.replans = 0; self.min_front = float("inf"); self._consec_near = 0
            self.outcome = None
        def update(self, front_min: float, blocked_step: bool) -> bool:
            self.steps += 1; self.min_front = min(self.min_front, float(front_min))
            if blocked_step: self.blocked_steps += 1
            self._consec_near = self._consec_near + 1 if front_min <= CRASH_DIST else 0
            return self._consec_near >= CRASH_STEPS
        def timed_out(self) -> bool: return (time.time() - self.t0) >= GOAL_TIMEOUT_SEC
        def add_replan(self, n=1): self.replans += int(n)
        def _summary_row(self):
            dt = max(0.0, time.time() - self.t0)
            return {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "id": self.uid, "goal": self.goal, "outcome": self.outcome or "unknown",
                    "sec": round(dt,2), "steps": self.steps, "blocked_steps": self.blocked_steps,
                    "replans": self.replans,
                    "min_front": round(self.min_front if np.isfinite(self.min_front) else 9999,3)}
    def _append_dwa_summary(row_dict):
        os.makedirs("outputs", exist_ok=True)
        path = os.path.join("outputs", "dwa_summary.csv")
        header = ["timestamp","id","goal","outcome","sec","steps","blocked_steps","replans","min_front"]
        write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header: w.writeheader()
            w.writerow(row_dict)

    # A* metrics log
    def _append_astar_metrics(row_dict):
        os.makedirs("outputs", exist_ok=True)
        path = os.path.join("outputs", "astar_metrics.csv")
        header = ["timestamp","id","goal","context","is_replan","ok","fail_reason",
                  "plan_time_ms","nodes_expanded","nodes_generated","open_list_max",
                  "path_len_m","path_cost_grid","w"]
        write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header: w.writeheader()
            w.writerow(row_dict)

    def _mk_astar_row(m, uid, goal_label, context):
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "id": uid or "anon",
            "goal": goal_label or "",
            "context": context,                      
            "is_replan": int(m.get("is_replan", 0)),
            "ok": int(m.get("ok", 0)),
            "fail_reason": m.get("fail_reason"),
            "plan_time_ms": round(float(m.get("plan_time_ms", 0.0)), 3),
            "nodes_expanded": int(m.get("nodes_expanded", 0)),
            "nodes_generated": int(m.get("nodes_generated", 0)),
            "open_list_max": int(m.get("open_list_max", 0)),
            "path_len_m": round(float(m.get("path_len_m", 0.0)), 3),
            "path_cost_grid": None if m.get("path_cost_grid") is None else round(float(m["path_cost_grid"]), 3),
            "w": float(m.get("w", 1.5)),
        }

    def _append_pose_row(ep_id, goal, t_sim, x, y):
        os.makedirs("outputs", exist_ok=True)
        path = os.path.join("outputs", "pose_log.csv")
        header = ["timestamp", "episode_id", "goal", "t_sim", "x", "y"]
        write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                        ep_id or "anon",
                        goal or "",
                        round(float(t_sim), 3),
                        float(x), float(y)])
                    
    # Dynamic painting 
    def paint_dynamic_only_from_lidar(occ_base, scan_LH, angles_h, max_range, x, y, yaw,
                                      res_coarse, radius_m=0.35, cone_deg=None,
                                      static_occ_hard=None, static_dt_m=None,
                                      static_tol_m=0.25):
        if cone_deg is None: cone_deg = FORWARD_CONE_DEG
        pts = build_lidar_obstacles_layers(scan_LH, angles_h, max_range, x, y, yaw,
                                           downsample_h=4, cone_deg=cone_deg)
        if pts.size == 0:
            return (np.empty((0,2), float), np.empty((0,2), float))

        Hc, Wc = occ_base.shape
        r_cells = max(1, int(radius_m / res_coarse))
        if static_occ_hard is None:
            static_occ_hard = np.zeros_like(occ_base, dtype=np.uint8)
        if static_dt_m is None:
            static_dt_m = np.full_like(occ_base, np.inf, dtype=np.float32)

        kept_dyn, rej_stat = [], []
        for (ox, oy) in pts:
            gx, gy = to_grid(ox, oy, res_coarse)
            if not (0 <= gx < Hc and 0 <= gy < Wc): continue
            if static_occ_hard[gx, gy] > 0 or (static_dt_m[gx, gy] < static_tol_m):
                rej_stat.append((ox, oy))
                continue
            kept_dyn.append((ox, oy))
            x0, x1 = max(0, gx - r_cells), min(Hc - 1, gx + r_cells)
            y0, y1 = max(0, gy - r_cells), min(Wc - 1, gy + r_cells)
            for ix in range(x0, x1 + 1):
                dx2 = (ix - gx) * (ix - gx)
                for iy in range(y0, y1 + 1):
                    if dx2 + (iy - gy) * (iy - gy) <= r_cells * r_cells:
                        occ_base[ix, iy] = max(occ_base[ix, iy], DYN_MARK_VALUE)

        return (np.array(kept_dyn, float) if kept_dyn else np.empty((0,2), float),
                np.array(rej_stat, float) if rej_stat else np.empty((0,2), float))

    def _dynamic_layers_for_planning(goal_gxy=None):
    
        dyn_bin = (dynamic_occ_coarse >= DYN_THRESH_BIN).astype(np.uint8)
        r_hard = max(1, int(DYN_HARD_INFLATE_M / RES_COARSE))
        if r_hard > 0:
            se = np.zeros((2*r_hard+1, 2*r_hard+1), dtype=np.uint8)
            yy, xx = np.ogrid[-r_hard:r_hard+1, -r_hard:r_hard+1]
            se[xx**2 + yy**2 <= r_hard**2] = 1
            dyn_hard = scipy.ndimage.binary_dilation(dyn_bin, structure=se).astype(np.uint8)
        else:
            dyn_hard = dyn_bin

        inv = 1 - dyn_bin
        dist_cells = scipy.ndimage.distance_transform_edt(inv)
        dist_m     = dist_cells * RES_COARSE
        soft = np.exp(-np.maximum(0.0, dist_m) / max(1e-6, DYN_COST_FALLOFF_M))
        soft[dist_m >= DYN_SOFT_INFLATE_M] = 0.0
        penalty_dyn = (soft * DYN_COST_SCALE).astype(np.float32)

        if goal_gxy is not None:
            gx, gy = goal_gxy
            if 0 <= gx < penalty_dyn.shape[0] and 0 <= gy < penalty_dyn.shape[1]:
                rg = max(1, int(DYN_GOAL_RELIEF_M / RES_COARSE))
                x0, x1 = max(0, gx-rg), min(penalty_dyn.shape[0]-1, gx+rg)
                y0, y1 = max(0, gy-rg), min(penalty_dyn.shape[1]-1, gy+rg)
                sub = penalty_dyn[x0:x1+1, y0:y1+1]
                fy, fx = np.ogrid[0:sub.shape[0], 0:sub.shape[1]]
                rr = np.hypot(fx - (gy - y0), fy - (gx - x0)) * RES_COARSE
                feather = np.clip(rr / DYN_GOAL_RELIEF_M, 0.0, 1.0)
                sub *= feather
                penalty_dyn[x0:x1+1, y0:y1+1] = sub

        learned_infl = learned_static_mask
        if LEARN_STATIC_INFLATE_M > 1e-6:
            learned_infl = scipy.ndimage.binary_dilation(learned_static_mask, structure=SE_LEARN_INFL).astype(np.uint8)

        occ_for_plan     = np.maximum(np.maximum(inflated_occ_static, learned_infl), dyn_hard).astype(np.uint8)
        penalty_for_plan = penalty_map_static.astype(np.float32) + penalty_dyn
        return occ_for_plan, penalty_for_plan

    def _path_intrudes_mask(path, start_idx, lookahead_m, corridor_m, mask) -> bool:
        if not path or start_idx >= len(path): return False
        r_corr = max(1, int(corridor_m / RES_COARSE))
        Hc, Wc = mask.shape
        acc_m = 0.0
        prev_w = to_world(path[start_idx][0], path[start_idx][1], RES_COARSE)
        for k in range(start_idx, len(path)):
            wx, wy = to_world(path[k][0], path[k][1], RES_COARSE)
            acc_m += math.hypot(wx - prev_w[0], wy - prev_w[1]); prev_w = (wx, wy)
            gx, gy = to_grid(wx, wy, RES_COARSE)
            if 0 <= gx < Hc and 0 <= gy < Wc:
                x0, x1 = max(0, gx - r_corr), min(Hc - 1, gx + r_corr)
                y0, y1 = max(0, gy - r_corr), min(Wc - 1, gy + r_corr)
                if np.any(mask[x0:x1+1, y0:y1+1] > 0): return True
            if acc_m >= lookahead_m: break
        return False

    def _is_obstruction_static_ahead(scan_LH, angles_h, max_range, x, y, yaw,
                                     cone_deg=60, static_tol_m=0.25) -> bool:
        pts = build_lidar_obstacles_layers(
            scan_LH, angles_h, max_range, x, y, yaw, downsample_h=3, cone_deg=cone_deg
        )
        if pts.size == 0: return False
        learned_infl = learned_static_mask
        if LEARN_STATIC_INFLATE_M > 1e-6:
            learned_infl = scipy.ndimage.binary_dilation(learned_static_mask, structure=SE_LEARN_INFL).astype(np.uint8)
        static_combined = np.maximum(inflated_occ_static, learned_infl)
        Hc, Wc = static_combined.shape
        for (ox, oy) in pts:
            gx, gy = to_grid(ox, oy, RES_COARSE)
            if 0 <= gx < Hc and 0 <= gy < Wc:
                if static_combined[gx, gy] > 0: return True
                if STATIC_DT_M[gx, gy] < static_tol_m: return True
        return False

    def plan_from_current_to(xy, *, is_replan=False, context="itinerary", goal_label=None):
        cx, cy = gps.getValues()[0], gps.getValues()[1]
        cgx, cgy = to_grid(cx, cy, RES_COARSE)
        gx,  gy  = to_grid(xy[0], xy[1], RES_COARSE)
        occ_for_plan, penalty_for_plan = _dynamic_layers_for_planning(goal_gxy=(gx, gy))
        force_goal_free(occ_for_plan, (gx, gy), max(1, int(0.25/RES_COARSE)))

        budget = globals().get("ASTAR_TIME_BUDGET_MS", None)

        path, m = weighted_a_star(
            (cgx, cgy), (gx, gy),
            occ_for_plan,
            penalty=penalty_for_plan,
            w=1.5,
            cell_size_m=RES_COARSE,
            is_replan=is_replan,
            time_budget_ms=budget
        )

        if m is None: m = {"ok": int(bool(path)), "is_replan": int(is_replan), "w": 1.5}

        _append_astar_metrics(_mk_astar_row(m, current_unique_id, goal_label or cur_target_label, context))
        return path, m

    path_step = 0; last_w = 0.0
    emerg_active = False; avoid_count = 0
    no_progress_steps = 0; prev_dist_to_carrot = None
    blocked_consec = 0; PAUSED = True; WAITING_USER_AT_WAYPOINT = False
    stall_replan_cooldown = 0
    goal_gx_c = 0
    goal_gy_c = 0

    robot_path = []; all_paths = []
    current_episode = None
    dyn_replan_cooldown = 0

    def waypoints_ready(): return len(waypoints) > 0

    visited, skipped = set(), set()
    def push_itinerary_state():
        labels = [label for (_x, _y, label) in waypoints] if waypoints else []
        desired_idx = -1
        if waypoints and 0 <= cur_idx < len(waypoints):
            if cur_target_label == waypoints[cur_idx][2]: desired_idx = cur_idx
        ui.set_state(itinerary=labels, current_idx=desired_idx,
                     visited_indices=sorted(list(visited)),
                     skipped_indices=sorted(list(skipped)))

    def hil_pause():
        nonlocal PAUSED
        PAUSED = True; left_motor.setVelocity(0.0); right_motor.setVelocity(0.0)
        ui.set_state(status="paused"); print("[CTRL] Paused.")

    def hil_resume():
        nonlocal PAUSED
        if WAITING_USER_AT_WAYPOINT: ui.set_state(toast="At waypoint. Press Next.")
        else: PAUSED = False; ui.set_state(status="running")
        print("[CTRL] Resume requested.")

    def plan_to_idx(idx):
        nonlocal path, path_step, cur_target_label, goal_gx_c, goal_gy_c, current_episode
        nonlocal LOITERING, PENDING_GOAL_WORLD, loiter_center, loiter_phase, replan_tick
        tx, ty, cur_target_label = waypoints[idx]
        ui.set_state(target=cur_target_label, current_idx=idx)
        goal_gx_c, goal_gy_c = to_grid(tx, ty, RES_COARSE)
        p, m = plan_from_current_to((tx, ty), is_replan=False, context="itinerary", goal_label=cur_target_label)
        if p:
            path = p; path_step = 0; all_paths.append(path)
            LOITERING = False; PENDING_GOAL_WORLD = None
            ui.set_state(toast=f"Navigating to {cur_target_label}")
            push_itinerary_state()
            current_episode = DwaEpisode(current_unique_id, cur_target_label)
            print(f"[CTRL] Planned to itinerary idx {idx}: {cur_target_label}")
            return True

        pos = gps.getValues(); cx, cy = pos[0], pos[1]
        LOITERING = True; PENDING_GOAL_WORLD = (tx, ty)
        loiter_center = (cx, cy); loiter_phase = 0.0; replan_tick = 0
        ui.set_state(toast=f"No path now. Searching locally for a route to {cur_target_label}…")
        print(f"[CTRL] No path to {cur_target_label}. Entering LOITER mode.")
        return False

    def begin_loiter_to(tx, ty, label_for_ui: str):
        nonlocal LOITERING, PENDING_GOAL_WORLD, loiter_center, loiter_phase, replan_tick, cur_target_label
        cur_target_label = label_for_ui
        ui.set_state(target=cur_target_label, current_idx=-1)
        pos = gps.getValues(); cx, cy = pos[0], pos[1]
        LOITERING = True; PENDING_GOAL_WORLD = (tx, ty)
        loiter_center = (cx, cy); loiter_phase = 0.0; replan_tick = 0
        ui.set_state(toast=f"No path now. Searching locally for a route to {cur_target_label}…")
        print(f"[CTRL] Begin LOITER toward “{label_for_ui}” at {tx:.2f},{ty:.2f}")

    def hil_next():
        nonlocal cur_idx, diverted_from_idx, RETURN_OFFER_IDX, AWAIT_RETURN_DECISION
        nonlocal WAITING_USER_AT_WAYPOINT, PAUSED
        if not waypoints_ready(): ui.set_state(toast="Enter a Unique ID first."); return
        if not WAITING_USER_AT_WAYPOINT: ui.set_state(toast="Next only works after ARRIVED."); return
        last_was_itinerary = (waypoints and 0 <= cur_idx < len(waypoints) and
                              cur_target_label == waypoints[cur_idx][2])
        if last_was_itinerary:
            RETURN_OFFER_IDX = None; AWAIT_RETURN_DECISION = False; cur_idx += 1
        if cur_idx >= len(waypoints):
            ui.set_state(status="completed", toast="Itinerary completed.", current_idx=-1)
            WAITING_USER_AT_WAYPOINT = False; PAUSED = True
            left_motor.setVelocity(0.0); right_motor.setVelocity(0.0); return
        if plan_to_idx(cur_idx):
            WAITING_USER_AT_WAYPOINT = False; PAUSED = False
            ui.set_state(status="running"); push_itinerary_state()

    def insert_waypoint(dx, dy, label, navigate_now=True):
        nonlocal waypoints, cur_idx, visited, skipped, PAUSED

        if waypoints and 0 <= cur_idx < len(waypoints):
            insert_at = (cur_idx + 1) if (cur_idx in visited) else cur_idx
        else:
            insert_at = len(waypoints)

        waypoints.insert(insert_at, (dx, dy, label))

        visited = { (i + 1 if i >= insert_at else i) for i in visited }
        skipped = { (i + 1 if i >= insert_at else i) for i in skipped }

        push_itinerary_state()

        if navigate_now:
            cur_idx = insert_at
            if plan_to_idx(cur_idx):
                PAUSED = False
                ui.set_state(status="running", target=label, current_idx=cur_idx,
                             toast=f"Navigating to {label}")
            else:
                begin_loiter_to(dx, dy, label)
                PAUSED = False
            push_itinerary_state()

    def hil_divert(category, tag):
        nonlocal diverted_from_idx, RETURN_OFFER_IDX, AWAIT_RETURN_DECISION
        nonlocal path, path_step, cur_target_label, PAUSED, current_episode
        if not waypoints_ready(): ui.set_state(toast="Enter a Unique ID first."); return
        pos = gps.getValues(); divert = nearest_by_category(pos[0], pos[1], category)
        if not divert: ui.set_state(toast=f"No points for category '{category}'."); return
        diverted_from_idx = cur_idx; RETURN_OFFER_IDX = diverted_from_idx; AWAIT_RETURN_DECISION = False
        dx, dy, name_key = divert
        label_for_log = f"({tag}) {name_key}"
        p, m = plan_from_current_to((dx, dy), is_replan=False, context="divert", goal_label=label_for_log)
        if p:
            path = p; path_step = 0; all_paths.append(path)
            cur_target_label = label_for_log
            ui.set_state(target=cur_target_label, current_idx=-1, toast=f"{tag.capitalize()} → {name_key}")
            push_itinerary_state()
            print(f"[CTRL] Diverting to {tag}: {name_key}")
            PAUSED = False; return
        begin_loiter_to(dx, dy, f"({tag}) {name_key}"); PAUSED = False

    def _is_current_target_itinerary():
        return (waypoints and 0 <= cur_idx < len(waypoints) and
                cur_target_label == waypoints[cur_idx][2])

    print("[INFO] Open http://localhost:8765 and enter a Unique ID.")

    # Main loop 
    try:
        while robot.step(TIME_STEP) != -1:
            if dyn_replan_cooldown > 0: dyn_replan_cooldown -= 1
            if stall_replan_cooldown > 0: stall_replan_cooldown -= 1

            # UI
            msg = ui.get_nowait()
            if isinstance(msg, dict):
                cmd, arg = msg.get("cmd",""), msg.get("arg")
                if cmd == "SET_ID":
                    left_motor.setVelocity(0.0); right_motor.setVelocity(0.0); PAUSED = True
                    new_id = (arg or "").strip()
                    if not new_id:
                        ui.set_state(toast="Please enter a Unique ID.")
                    else:
                        current_unique_id = new_id; ui.set_state(current_id=current_unique_id)
                        names = load_names_from_csv(CSV_FILE, current_unique_id)
                        waypoints = names_to_waypoints(names); cur_idx = 0
                        visited.clear(); skipped.clear()
                        RETURN_OFFER_IDX = None; AWAIT_RETURN_DECISION = False
                        if not waypoints:
                            ui.set_state(status="idle", target="", itinerary=[],
                                         toast=f"No valid itinerary for '{current_unique_id}'."); path = []
                            print(f"[CTRL] No itinerary for ID {current_unique_id}")
                        else:
                            labels = [label for (_,_,label) in waypoints]
                            tx, ty, cur_target_label = waypoints[cur_idx]
                            ui.set_state(target=cur_target_label, toast=f"Loaded itinerary for '{current_unique_id}'",
                                         itinerary=labels, current_idx=cur_idx)
                            push_itinerary_state()
                            p, m = plan_from_current_to((tx, ty), is_replan=False, context="initial", goal_label=cur_target_label)
                            if p:
                                path = p; path_step = 0; all_paths.append(path)
                                goal_gx_c, goal_gy_c = to_grid(tx, ty, RES_COARSE)
                                current_episode = DwaEpisode(current_unique_id, cur_target_label)
                                PAUSED = False; ui.set_state(status="running", toast=f"Navigating to {cur_target_label}")
                                print(f"[CTRL] Planned to first waypoint: {cur_target_label}")
                            else:
                                begin_loiter_to(tx, ty, cur_target_label); PAUSED = False
                                ui.set_state(status="running", toast=f"Navigating to {cur_target_label}")
                                push_itinerary_state()

                elif cmd == "STOP": hil_pause()
                elif cmd == "RESUME": hil_resume()
                elif cmd == "SKIP":
                    if _is_current_target_itinerary():
                        if waypoints and 0 <= cur_idx < len(waypoints): skipped.add(cur_idx)
                        ui.set_state(toast=f"Skipped {cur_target_label}"); cur_idx += 1
                        if cur_idx >= len(waypoints):
                            PAUSED = True; left_motor.setVelocity(0.0); right_motor.setVelocity(0.0)
                            ui.set_state(status="completed", current_idx=-1)
                        else:
                            if plan_to_idx(cur_idx): PAUSED = False; ui.set_state(status="running")
                        push_itinerary_state()
                    else:
                        ui.set_state(toast=f"Cancelled {cur_target_label}")
                        if waypoints and 0 <= cur_idx < len(waypoints):
                            if plan_to_idx(cur_idx): PAUSED = False; ui.set_state(status="running")
                        else: PAUSED = True
                elif cmd == "NEXT":
                    if not WAITING_USER_AT_WAYPOINT:
                        ui.set_state(toast="Next only works after ARRIVED.")
                    else:
                        if _is_current_target_itinerary(): visited.add(cur_idx); cur_idx += 1
                        WAITING_USER_AT_WAYPOINT = False; PAUSED = False
                        if cur_idx >= len(waypoints):
                            left_motor.setVelocity(0.0); right_motor.setVelocity(0.0)
                            PAUSED = True
                            ui.set_state(status="completed", current_idx=-1, toast="Itinerary completed.")
                        else:
                            if plan_to_idx(cur_idx):
                                ui.set_state(status="running", toast="Moving to next stop")
                        push_itinerary_state()

                elif cmd in ["FOOD", "BREAK"]:
                    category = "food" if cmd == "FOOD" else "relax"
                    tag = "Food stop" if cmd == "FOOD" else "Break stop"
                    pos = gps.getValues()
                    divert = nearest_by_category(pos[0], pos[1], category)
                    if not divert:
                        ui.set_state(toast=f"No points for category '{category}'.")
                    else:
                        dx, dy, name_key = divert
                        label = f"{tag} → {pretty_name(name_key)}"
                        insert_waypoint(dx, dy, label, navigate_now=True)

                elif cmd == "SET_DEST_NAME":
                    name = (arg or "").strip(); key = KEY_BY_NAME.get(name)
                    if not key:
                        ui.set_state(toast="Unknown destination.")
                    else:
                        dx, dy = NAMED_POINTS[key]["xy"]
                        p, m = plan_from_current_to((dx, dy), is_replan=False, context="named", goal_label=name)
                        if p:
                            path = p; path_step = 0; all_paths.append(path)
                            cur_target_label = name
                            ui.set_state(status="running", target=cur_target_label, current_idx=-1,
                                         toast=f"Navigating to {cur_target_label}")
                            push_itinerary_state()
                            current_episode = DwaEpisode(current_unique_id, cur_target_label)
                            print(f"[CTRL] Planned to named destination: {name}")
                            PAUSED = False
                        else:
                            begin_loiter_to(dx, dy, name); PAUSED = False

                elif cmd == "ADD_DEST_NAME":
                    name = (arg or "").strip()
                    key = KEY_BY_NAME.get(name)
                    if not key:
                        ui.set_state(toast="Unknown destination.")
                    else:
                        dx, dy = NAMED_POINTS[key]["xy"]
                        insert_waypoint(dx, dy, name, navigate_now=True)

                elif cmd == "SET_DEST":
                    try:
                        idx = int(arg)
                        if 0 <= idx < len(waypoints):
                            cur_idx = idx
                            if plan_to_idx(cur_idx):
                                PAUSED = False
                                ui.set_state(status="running", target=cur_target_label, current_idx=cur_idx,
                                             toast=f"Navigating to {cur_target_label}")
                                push_itinerary_state()
                        else:
                            ui.set_state(toast="Invalid destination index.")
                    except Exception:
                        ui.set_state(toast="Invalid destination request.")

            if PAUSED or (not path and not LOITERING):
                pos = gps.getValues(); x, y = pos[0], pos[1]; yaw = imu.getRollPitchYaw()[2]
                #  Pose logging 
                t_sim = robot.getTime()
                ep_id = (current_episode.uid if ('current_episode' in locals() and current_episode) else (current_unique_id or "anon"))
                ep_goal = (current_episode.goal if ('current_episode' in locals() and current_episode) else (cur_target_label or ""))
                _append_pose_row(ep_id, ep_goal, t_sim, x, y)
            
                scan_LH = read_scan_layers(lidar)
                slam.update_from_scan((x, y, yaw), np.min(scan_LH, axis=0), H, fov, max_range)
                left_motor.setVelocity(0.0); right_motor.setVelocity(0.0)
                continue

            # Pose & SLAM update
            pos = gps.getValues(); x, y = pos[0], pos[1]; yaw = imu.getRollPitchYaw()[2]
            
            # Pose logging (during motion)
            t_sim = robot.getTime()
            ep_id = (current_episode.uid if ('current_episode' in locals() and current_episode) else (current_unique_id or "anon"))
            ep_goal = (current_episode.goal if ('current_episode' in locals() and current_episode) else (cur_target_label or ""))
            _append_pose_row(ep_id, ep_goal, t_sim, x, y)
            
            robot_path.append((x, y))
            scan_LH = read_scan_layers(lidar)
            slam.update_from_scan((x, y, yaw), np.min(scan_LH, axis=0), H, fov, max_range)
            
            scan_min_h_now = np.min(scan_LH, axis=0)
            front_mask_25 = np.abs(angles_h) < math.radians(25)
            front_min_25_live = float(np.min(scan_min_h_now[front_mask_25])) if np.any(front_mask_25) else float("inf")
            widen_cone = front_min_25_live < (FORWARD_CLEAR_DIST + 0.10)

            # Update dynamic occupancy 
            dynamic_occ_coarse *= (1.0 - DYN_DECAY_PER_STEP)
            dynamic_occ_coarse = np.clip(dynamic_occ_coarse, 0.0, 1.0)
            widen_deg = max(FORWARD_CONE_DEG, 140) if widen_cone else FORWARD_CONE_DEG

            # combine original static with learned-static 
            learned_infl_for_filter = learned_static_mask
            if LEARN_STATIC_INFLATE_M > 1e-6:
                learned_infl_for_filter = scipy.ndimage.binary_dilation(
                    learned_static_mask, structure=SE_LEARN_INFL
                ).astype(np.uint8)
            static_for_filter = np.maximum(inflated_occ_static, learned_infl_for_filter)

            obstacles_dyn_xy, obstacles_stat_xy = paint_dynamic_only_from_lidar(
                dynamic_occ_coarse, scan_LH, angles_h, max_range, x, y, yaw,
                RES_COARSE, radius_m=0.35, cone_deg=widen_deg,
                static_occ_hard=inflated_occ_static,
                static_dt_m=STATIC_DT_M,
                static_tol_m=STATIC_NEAR_TOL_M
            )

            if path:
                goal_world = to_world(path[-1][0], path[-1][1], RES_COARSE)
                dist_goal = math.hypot(goal_world[0]-x, goal_world[1]-y)
                if dist_goal <= GOAL_CAPTURE_DIST:
                    left_motor.setVelocity(0.0); right_motor.setVelocity(0.0)
                    WAITING_USER_AT_WAYPOINT = True; PAUSED = True
                    if waypoints and 0 <= cur_idx < len(waypoints):
                        if cur_target_label == waypoints[cur_idx][2]: visited.add(cur_idx)
                    path = []; path_step = 0
                    LOITERING = False; PENDING_GOAL_WORLD = None
                    blocked_consec = 0; no_progress_steps = 0; prev_dist_to_carrot = None
                    dyn_replan_cooldown = 0
                    ui.set_state(status="arrived", toast=f"Arrived: {cur_target_label}")
                    push_itinerary_state(); print(f"[CTRL] Arrived at {cur_target_label}")

                    if 'current_episode' in locals() and current_episode:
                        current_episode.outcome = "success"; _append_dwa_summary(current_episode._summary_row())
                        current_episode = None
                    continue

                near_goal = dist_goal <= NEAR_GOAL_RADIUS
                if near_goal:
                    tx, ty = goal_world
                else:
                    if path_step >= len(path): path_step = len(path)-1
                    idx_ = min(path_step + LOOKAHEAD, len(path) - 1)
                    tgt = path[idx_]; tx, ty = to_world(tgt[0], tgt[1], RES_COARSE)
                    if np.hypot(tx - x, ty - y) < WAYPOINT_RADIUS:
                        path_step += 1; prev_dist_to_carrot = None; no_progress_steps = 0

                blocked_ahead = False
                combined_static_for_chk = np.maximum(inflated_occ_static, learned_static_mask)
                for gx, gy in path[path_step:min(path_step + BLOCK_CHECK_AHEAD, len(path))]:
                    if combined_static_for_chk[gx, gy] > 0: blocked_ahead = True; break
                bearing = math.atan2(ty - y, tx - x)
                ang_err = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
                if (front_min_25_live < (FORWARD_CLEAR_DIST - 0.02)) and (abs(ang_err) < ANGLE_ALIGN_RAD):
                    blocked_ahead = True

            else:
                near_goal = False; blocked_ahead = False
                if loiter_center is None: loiter_center = (x, y)
                cx, cy = loiter_center; cx = 0.98*cx + 0.02*x; cy = 0.98*cy + 0.02*y
                loiter_center = (cx, cy); loiter_phase += LOITER_ANG_RATE * (TIME_STEP/1000.0)
                tx = cx + LOITER_RADIUS * math.cos(loiter_phase)
                ty = cy + LOITER_RADIUS * math.sin(loiter_phase)

            # DWA local control
            v_cmd, w_cmd, blocked_this_step, front_min, dbg_info = evaluate_dwa(
                x, y, yaw, tx, ty, angles_h, scan_LH, max_range, last_w, near_goal, blocked_ahead,
                res_coarse=RES_COARSE,
                static_occ_hard=np.maximum(inflated_occ_static, learned_static_mask),
                static_dt_m=STATIC_DT_M,
                static_tol_m=STATIC_NEAR_TOL_M,
            )

            if LEARN_STATIC_ENABLE:
                dyn_bin_now = (dynamic_occ_coarse >= DYN_THRESH_BIN).astype(np.uint8)
                learn_candidates = np.logical_and(dyn_bin_now > 0, inflated_occ_static == 0)
                moving_now = False
                if isinstance(dbg_info, dict):
                    moving_now = dbg_info.get("max_track_speed", 0.0) >= LEARN_STATIC_VEL_THRESH_MS

                if not moving_now:
                    inc = learn_candidates.astype(np.uint8)
                    persist_counter = np.where(
                        inc > 0,
                        np.minimum(persist_counter + 1, np.iinfo(np.uint16).max),
                        0 if LEARN_STATIC_DECAY_PER_STEP <= 0 else
                        np.maximum(persist_counter - int(LEARN_STATIC_DECAY_PER_STEP), 0)
                    )
                    ready = (persist_counter >= LEARN_STATIC_PERSIST_STEPS).astype(np.uint8)
                    ready = scipy.ndimage.binary_opening(ready, structure=SE_LEARN_CLOSE).astype(np.uint8)
                    labels, n = scipy.ndimage.label(ready)
                    if n > 0:
                        sizes = np.bincount(labels.ravel())
                        MIN_LEARN_AREA_CELLS = globals().get("MIN_LEARN_AREA_CELLS", 12)
                        small = np.isin(labels, np.where(sizes < MIN_LEARN_AREA_CELLS)[0])
                        ready[small] = 0
                    newly_learned = ready
                    if newly_learned.any():
                        closed = scipy.ndimage.binary_closing(newly_learned, structure=SE_LEARN_CLOSE).astype(np.uint8)
                        learned_static_mask = np.maximum(learned_static_mask, closed)
                        dynamic_occ_coarse[learned_static_mask > 0] = 0.0
                else:
                    persist_counter = np.maximum(persist_counter - 1, 0)

            if path and not near_goal and dyn_replan_cooldown == 0:
                try:
                    learned_infl_for_corridor = learned_static_mask
                    if LEARN_STATIC_INFLATE_M > 1e-6:
                        learned_infl_for_corridor = scipy.ndimage.binary_dilation(
                            learned_static_mask, structure=SE_LEARN_INFL
                        ).astype(np.uint8)

                    static_intrude = _path_intrudes_mask(
                        path, path_step, DYN_REPLAN_LOOKAHEAD_M, DYN_REPLAN_CORRIDOR_M,
                        learned_infl_for_corridor
                    )

                    dyn_bin = (dynamic_occ_coarse >= DYN_THRESH_BIN).astype(np.uint8)
                    r_hard = max(1, int(DYN_HARD_INFLATE_M / RES_COARSE))
                    if r_hard > 0:
                        se = np.zeros((2*r_hard+1, 2*r_hard+1), dtype=np.uint8)
                        yy, xx = np.ogrid[-r_hard:r_hard+1, -r_hard:r_hard+1]
                        se[xx**2 + yy**2 <= r_hard**2] = 1
                        dyn_hard_now = scipy.ndimage.binary_dilation(dyn_bin, structure=se).astype(np.uint8)
                    else:
                        dyn_hard_now = dyn_bin
                    dynamic_intrude = _path_intrudes_mask(
                        path, path_step, DYN_REPLAN_LOOKAHEAD_M, DYN_REPLAN_CORRIDOR_M, dyn_hard_now
                    )

                    if static_intrude:
                        gx_goal, gy_goal = path[-1][0], path[-1][1]
                        tx_goal, ty_goal = to_world(gx_goal, gy_goal, RES_COARSE)
                        new_path, m = plan_from_current_to((tx_goal, ty_goal), is_replan=True,
                                                           context="proactive", goal_label=cur_target_label)
                        if new_path:
                            path = new_path; path_step = 0; all_paths.append(path)
                            if current_episode: current_episode.add_replan(1)
                            ui.set_state(toast="Learned-static ahead → global replan.")
                            print("[CTRL] Proactive learned-static replan.")
                            dyn_replan_cooldown = DYN_REPLAN_COOLDOWN_STEPS
                    elif dynamic_intrude:
                        pass
                except Exception as e:
                    print(f"[CTRL] Proactive replan check failed: {e}")

            if BLOCKED_REQUIRE_BOTH:
                blocked_consec = blocked_consec + 1 if (blocked_this_step and blocked_ahead) else 0
            else:
                blocked_consec = blocked_consec + 1 if (blocked_this_step or blocked_ahead) else 0

            if blocked_consec >= BLOCKED_CONSECUTIVE_FOR_REPLAN and path:
                is_static_block = _is_obstruction_static_ahead(
                    scan_LH, angles_h, max_range, x, y, yaw,
                    cone_deg=max(60, FORWARD_CONE_DEG), static_tol_m=STATIC_NEAR_TOL_M
                )
                if is_static_block:
                    gx_goal, gy_goal = path[-1][0], path[-1][1]
                    tx_goal, ty_goal = to_world(gx_goal, gy_goal, RES_COARSE)
                    new_path, m = plan_from_current_to((tx_goal, ty_goal), is_replan=True,
                                                       context="blocked", goal_label=cur_target_label)
                    if new_path:
                        path = new_path; path_step = 0; all_paths.append(path)
                        if current_episode: current_episode.add_replan(1)
                        ui.set_state(toast="Static block → global replan."); print("[CTRL] Blocked-by-static replan.")
                    else:
                        ui.set_state(toast="Static block but replan failed."); print("[CTRL] Static replan failed.")
                blocked_consec = 0

            if path and not near_goal:
                dist_to_carrot = math.hypot(tx - x, ty - y)
                min_step_progress = max(PROGRESS_EPSILON, PROGRESS_RATE_MPS_MIN * (TIME_STEP / 1000.0))
                bearing = math.atan2(ty - y, tx - x)
                ang_err = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
                if prev_dist_to_carrot is None:
                    prev_dist_to_carrot = dist_to_carrot
                    no_progress_steps = 0
                else:
                    improved = (prev_dist_to_carrot - dist_to_carrot) >= min_step_progress
                    if abs(ang_err) > ANGLE_ALIGN_RAD:
                        no_progress_steps = 0
                    else:
                        no_progress_steps = 0 if improved else (no_progress_steps + 1)
                    prev_dist_to_carrot = dist_to_carrot

            if path and (no_progress_steps > STALL_STEPS_REPLAN) and (stall_replan_cooldown == 0) and (dist_to_carrot > STALL_MIN_DIST_M):
                gx_goal, gy_goal = path[-1][0], path[-1][1]
                tx_goal, ty_goal = to_world(gx_goal, gy_goal, RES_COARSE)
                new_path, m = plan_from_current_to((tx_goal, ty_goal), is_replan=True,
                                                   context="stall", goal_label=cur_target_label)
                if new_path:
                    path = new_path; path_step = 0; all_paths.append(path)
                    if current_episode: current_episode.add_replan(1)
                    ui.set_state(toast="Stall replan succeeded."); print("[CTRL] Stall replan succeeded.")
                    stall_replan_cooldown = STALL_REPLAN_COOLDOWN_STEPS
                else:
                    ui.set_state(toast="Stall replan failed."); print("[CTRL] Stall replan failed.")
                    stall_replan_cooldown = STALL_REPLAN_COOLDOWN_STEPS
                no_progress_steps = 0; prev_dist_to_carrot = None

            if current_episode:
                crash_now = current_episode.update(front_min=front_min, blocked_step=blocked_this_step)
                if crash_now:
                    current_episode.outcome = "crash"
                    left_motor.setVelocity(0.0); right_motor.setVelocity(0.0)
                    ui.set_state(status="paused", toast=f"CRASH near {cur_target_label}")
                    _append_dwa_summary(current_episode._summary_row()); current_episode = None; PAUSED = True
                    continue
                if current_episode.timed_out():
                    current_episode.outcome = "timeout"
                    left_motor.setVelocity(0.0); right_motor.setVelocity(0.0)
                    ui.set_state(status="paused", toast=f"TIMEOUT at {cur_target_label}")
                    _append_dwa_summary(current_episode._summary_row()); current_episode = None; PAUSED = True
                    continue

            L, R = vw_to_wheels(v_cmd, w_cmd)
            left_motor.setVelocity(L); right_motor.setVelocity(R)
            last_w = w_cmd

    finally:
        try:
            left_motor.setVelocity(0.0); right_motor.setVelocity(0.0)
        except Exception:
            pass

if __name__ == "__main__":
    main()
    main()