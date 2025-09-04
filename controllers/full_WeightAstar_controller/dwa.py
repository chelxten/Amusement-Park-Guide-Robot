import numpy as np, math, time
from config import (
    AXLE_LEN, MAX_SPEED, V_MAX, W_MAX, V_GRID, W_GRID, DT_PRED,
    FRONT_ARC_DEG, FORWARD_CONE_DEG, FORWARD_CLEAR_DIST, OBSTACLE_CLEAR_MIN,
    W_GOAL, W_CLEAR, W_SPEED, W_SMOOTH, W_HEAD, W_CARROT_PULL, W_GOAL_DIST
)

try:
    from config import TIME_STEP, TURN_IN_PLACE_W, TURN_SLOW_W, V_TURN_SLOW
except Exception:
    TIME_STEP = 32  
    TURN_IN_PLACE_W = 0.8
    TURN_SLOW_W     = 0.5
    V_TURN_SLOW     = 0.12

try:
    from config import HEADING_DEADBAND_DEG, W_DEADBAND
except Exception:
    HEADING_DEADBAND_DEG = 10.0
    W_DEADBAND = 0.10

# Human modeling 
HUMAN_RADIUS      = globals().get("HUMAN_RADIUS", 0.60)   
HUMAN_MARGIN      = globals().get("HUMAN_MARGIN", 0.30)   
HUMAN_SIGMA_V     = globals().get("HUMAN_SIGMA_V", 0.50)  
V_OBS_MAX         = globals().get("V_OBS_MAX", 1.80)      

CLUSTER_EPS       = globals().get("CLUSTER_EPS", 0.60)
CLUSTER_MIN_PTS   = globals().get("CLUSTER_MIN_PTS", 2)
ASSOC_DIST        = globals().get("ASSOC_DIST", 1.00)
VEL_EMA_ALPHA     = globals().get("VEL_EMA_ALPHA", 0.4)
TRACK_FORGET_SEC  = globals().get("TRACK_FORGET_SEC", 1.2)
MIN_TRACK_SPEED   = globals().get("MIN_TRACK_SPEED", 0.25) 
MIN_TRACK_AGE     = globals().get("MIN_TRACK_AGE", 0.20)    

# Pedestrian risk scoring weights
W_PED_CLEAR       = globals().get("W_PED_CLEAR", 10.0)   
W_PED_TTC         = globals().get("W_PED_TTC", 3.0)     
TTC_MIN_CLAMP     = globals().get("TTC_MIN_CLAMP", 0.25)

def vw_to_wheels(v, w, axle=AXLE_LEN):
    left  = v - w * axle / 2.0
    right = v + w * axle / 2.0
    m = max(1.0, max(abs(left), abs(right)) / MAX_SPEED)
    return np.clip(left / m, -MAX_SPEED, MAX_SPEED), np.clip(right / m, -MAX_SPEED, MAX_SPEED)


def read_scan_layers(lidar):
    H = lidar.getHorizontalResolution()
    L = lidar.getNumberOfLayers() if hasattr(lidar, "getNumberOfLayers") else 1
    raw = np.array(lidar.getRangeImage())
    if raw.size != H*L:
        raw = np.nan_to_num(raw, nan=np.inf, posinf=np.inf, neginf=np.inf)
        return raw.reshape(1, -1)
    raw = raw.reshape(L, H)
    raw = np.nan_to_num(raw, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return raw


def build_lidar_obstacles_layers(scan_LH, angles_h, max_range, x, y, yaw,
                                 downsample_h=6, cone_deg=FORWARD_CONE_DEG):

    cone = math.radians(cone_deg) / 2.0
    L, H = scan_LH.shape
    pts = []
    for j in range(0, H, downsample_h):
        a = angles_h[j]
        if abs(a) > cone: continue
        r = float(np.min(scan_LH[:, j]))
        if not np.isfinite(r) or r <= 0.05 or r > max_range: continue
        wx = x + r * math.cos(yaw + a)
        wy = y + r * math.sin(yaw + a)
        pts.append((wx, wy))
    return np.array(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)


def filter_points_near_line(pts, p0, p1, max_lateral=0.6):
    if pts.size == 0: return pts
    v = np.array(p1) - np.array(p0)
    v2 = float(np.dot(v, v)) + 1e-9
    w = pts - np.array(p0)
    t = (w @ v) / v2
    t = np.clip(t, 0.0, 1.0)
    proj = np.array(p0) + t[:, None] * v
    lateral = np.hypot(*(pts - proj).T)
    return pts[lateral < max_lateral]

def _cluster_points(pts, eps=CLUSTER_EPS, min_pts=CLUSTER_MIN_PTS):
    if pts.size == 0: return []
    pts = np.asarray(pts)
    N = pts.shape[0]
    unvis = set(range(N))
    clusters = []
    while unvis:
        i = unvis.pop()
        cluster = [i]
        seed = [i]
        while seed:
            j = seed.pop()
            d = np.hypot(*(pts[list(unvis)] - pts[j]).T)
            nbr_ids = [k for k, dd in zip(list(unvis), d) if dd <= eps]
            for k in nbr_ids:
                unvis.remove(k); seed.append(k); cluster.append(k)
        if len(cluster) >= min_pts:
            clusters.append(pts[cluster].mean(axis=0))
    return clusters

class _Track:
    __slots__ = ("x","y","vx","vy","t_last","t_birth")
    def __init__(self, x, y, t):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = 0.0, 0.0
        self.t_last = t
        self.t_birth = t

    def age(self, tnow): return max(0.0, tnow - self.t_birth)

    def update(self, x, y, t):
        dt = max(1e-3, t - self.t_last)
        mx, my = (x - self.x) / dt, (y - self.y) / dt
        self.vx = (1.0 - VEL_EMA_ALPHA) * self.vx + VEL_EMA_ALPHA * mx
        self.vy = (1.0 - VEL_EMA_ALPHA) * self.vy + VEL_EMA_ALPHA * my
        v = math.hypot(self.vx, self.vy)
        if v > V_OBS_MAX and v > 1e-6:
            s = V_OBS_MAX / v
            self.vx *= s; self.vy *= s
        self.x, self.y = float(x), float(y)
        self.t_last = t

    def predict(self, dt):
        return (self.x + self.vx * dt, self.y + self.vy * dt)


class _ObstacleTracker:
    def __init__(self):
        self.tracks = []
        self.t_sim = 0.0

    def tick(self): self.t_sim += TIME_STEP / 1000.0
    def maintain(self): self.tracks = [tr for tr in self.tracks if (self.t_sim - tr.t_last) <= TRACK_FORGET_SEC]

    def update_with_centroids(self, cents):
        self.tick(); self.maintain()
        used = set()
        for tr in self.tracks:
            best, bestd, besti = None, 1e9, -1
            for i, c in enumerate(cents):
                if i in used: continue
                d = math.hypot(c[0] - tr.x, c[1] - tr.y)
                if d < bestd: bestd, best, besti = d, c, i
            if best is not None and bestd <= ASSOC_DIST:
                tr.update(best[0], best[1], self.t_sim)
                used.add(besti)
        for i, c in enumerate(cents):
            if i not in used:
                self.tracks.append(_Track(c[0], c[1], self.t_sim))

    def predicted_sequences(self, ts):
        out = []
        for tr in self.tracks:
            out.append([tr.predict(t) for t in ts])
        return out


def _deadband(val, tol): return 0.0 if val <= tol else (val - tol)
_TRACKER = _ObstacleTracker()

def evaluate_dwa(x, y, yaw, tx, ty, angles_h, scan_LH, max_range,
                 last_w, near_goal, blocked_ahead,
                 res_coarse=None, static_occ_hard=None, static_dt_m=None, static_tol_m=0.25):

    front_mask    = np.abs(angles_h) < math.radians(FRONT_ARC_DEG / 2)
    front_mask_25 = np.abs(angles_h) < math.radians(25)
    front_min     = np.min(scan_LH[:, front_mask])    if front_mask.any()    else float('inf')
    front_min_25  = np.min(scan_LH[:, front_mask_25]) if front_mask_25.any() else float('inf')

    forward_bias = (front_min_25 > FORWARD_CLEAR_DIST) and (not near_goal)
    cone_deg = FORWARD_CONE_DEG; down_h = 6
    if blocked_ahead or front_min_25 < (FORWARD_CLEAR_DIST + 0.10):
        cone_deg = max(cone_deg, 140); down_h = 4

    obstacles = build_lidar_obstacles_layers(scan_LH, angles_h, max_range, x, y, yaw,
                                             downsample_h=down_h, cone_deg=cone_deg)

    if (obstacles.size > 0 and res_coarse is not None and
        static_occ_hard is not None and static_dt_m is not None):
        Hc, Wc = static_occ_hard.shape
        keep = []
        for (ox, oy) in obstacles:
            gx = int(np.floor((ox + 0.5*res_coarse) / res_coarse)) 
            gy = int(np.floor((oy + 0.5*res_coarse) / res_coarse))
            if 0 <= gx < Hc and 0 <= gy < Wc:
                if static_occ_hard[gx, gy] > 0
                    continue
                if static_dt_m[gx, gy] < float(static_tol_m):
                    continue
            keep.append((ox, oy))
        obstacles = np.array(keep, dtype=float) if keep else np.empty((0, 2), dtype=float)

    if obstacles.size and front_min_25 > (FORWARD_CLEAR_DIST + 0.15) and not near_goal:
        obstacles = filter_points_near_line(obstacles, (x, y), (tx, ty), max_lateral=1.2)

    cents = _cluster_points(obstacles, eps=CLUSTER_EPS, min_pts=CLUSTER_MIN_PTS) if obstacles.size else []
    _TRACKER.update_with_centroids(cents)

    dyn_tracks = []
    max_track_speed = 0.0
    if dyn_tracks:
        max_track_speed = max(math.hypot(tr.vx, tr.vy) for tr in dyn_tracks)
        
    if _TRACKER.tracks:
        for tr in _TRACKER.tracks:
            v_tr = math.hypot(tr.vx, tr.vy)
            if v_tr < MIN_TRACK_SPEED: continue
            if tr.age(_TRACKER.t_sim) < MIN_TRACK_AGE: continue
            if static_dt_m is not None and res_coarse is not None:
                gx = int(np.floor((tr.x + 0.5*res_coarse) / res_coarse))
                gy = int(np.floor((tr.y + 0.5*res_coarse) / res_coarse))
                Hc, Wc = static_dt_m.shape
                if 0 <= gx < Hc and 0 <= gy < Wc:
                    if static_dt_m[gx, gy] < float(static_tol_m):
                        continue 
            dyn_tracks.append(tr)

    N_SAMPLES = 7
    ts = np.linspace(0.0, DT_PRED, N_SAMPLES)

    def rollout_pose(v, w, t):
        if abs(w) < 1e-4:
            return (x + v * math.cos(yaw) * t, y + v * math.sin(yaw) * t)
        R = v / max(1e-6, w)
        th = yaw + w * t
        cx = x - R * math.sin(yaw)
        cy = y + R * math.cos(yaw)
        px = cx + R * math.sin(th)
        py = cy - R * math.cos(th)
        return (px, py)

    v_cap = V_MAX
    if front_min < 0.60: v_cap = 0.6 * V_MAX
    if front_min < 0.45 or blocked_ahead: v_cap = 0.35 * V_MAX
    if near_goal: v_cap = min(v_cap, 0.60 * V_MAX)

    goal_angle = math.atan2(ty - y, tx - x)
    HEADING_DEADBAND_RAD = math.radians(HEADING_DEADBAND_DEG)

    best_score = -float('inf'); best_cmd = (0.0, 0.0)
    
    blocked = False
    if best_score == -float('inf'):
        scan_min_h_now = np.min(scan_LH, axis=0)
        best_idx = int(np.argmax(scan_min_h_now))
        best_angle = angles_h[best_idx]
        w_cmd = np.clip(2.2 * best_angle, -W_MAX, W_MAX)
        v_cmd = 0.0
        blocked = True
    else:
        v_cmd, w_cmd = best_cmd
        if v_cmd < 0.15 and abs(w_cmd) < 0.35 and front_min_25 > (OBSTACLE_CLEAR_MIN + 0.25):
            v_cmd = 0.18
    
    HEADING_IN_PLACE_DEG = 40; HEADING_SLOW_DEG = 25
    heading_err = abs((goal_angle - yaw + math.pi) % (2*math.pi) - math.pi)
    if abs(w_cmd) >= TURN_IN_PLACE_W and heading_err > math.radians(HEADING_IN_PLACE_DEG):
        v_cmd = 0.0
    elif abs(w_cmd) >= TURN_SLOW_W and heading_err > math.radians(HEADING_SLOW_DEG):
        v_cmd = min(v_cmd, V_TURN_SLOW)
    else:
        KAPPA_MAX = 3.0
        v_eps = max(v_cmd, 0.05)
        if abs(w_cmd) / v_eps > KAPPA_MAX:
            w_cmd = math.copysign(KAPPA_MAX * v_eps, w_cmd)
    
    dbg_info = {
        "num_tracks": len(_TRACKER.tracks),
        "max_track_speed": float(max_track_speed),
        "min_ttc": None,
    }

    
    for v in V_GRID:
        v = min(v, v_cap)
        for w in W_GRID:
            w = float(np.clip(w, -W_MAX, W_MAX))
            if forward_bias and obstacles.size == 0:
                ang_now = abs((goal_angle - yaw + math.pi) % (2*math.pi) - math.pi)
                if ang_now < math.radians(15): 
                    w = float(np.clip(w, -0.5, 0.5))

            px_end, py_end = rollout_pose(v, w, DT_PRED)
            if obstacles.size > 0:
                min_clear_static = float(np.min(np.hypot(obstacles[:,0]-px_end, obstacles[:,1]-py_end)))
            else:
                min_clear_static = max_range
            if min_clear_static <= OBSTACLE_CLEAR_MIN:
                continue

            ped_block = False
            min_ped_clear = float('inf')
            min_ped_ttc   = 1e9

            if dyn_tracks:
                rob_xy = np.array([rollout_pose(v, w, t) for t in ts])

                for tr in dyn_tracks:

                    ped_xy = np.array([tr.predict(t) for t in ts])

                    v_tr = min(V_OBS_MAX, math.hypot(tr.vx, tr.vy))

                    R_t = HUMAN_RADIUS + HUMAN_MARGIN + (HUMAN_SIGMA_V + 0.35*v_tr) * ts

                    d = np.hypot(*(rob_xy - ped_xy).T)  

                    hit_idx = np.where(d < R_t)[0]
                    if hit_idx.size > 0:
                        ped_block = True

                        k = int(hit_idx[0])
                        ttc = max(0.0, ts[k])
                        min_ped_ttc = min(min_ped_ttc, ttc if ttc > 1e-3 else 0.0)

                    min_ped_clear = min(min_ped_clear, float(np.min(d - R_t)))

                if ped_block:

                    continue

                if np.isfinite(min_ped_clear):
                    min_ped_clear = max(-2.0, min_ped_clear) 
                if min_ped_ttc < 1e9 and min_ped_ttc > 1e-3:
                    ped_ttc_pen = W_PED_TTC * (1.0 / max(TTC_MIN_CLAMP, min_ped_ttc))
                else:
                    ped_ttc_pen = 0.0
            else:
                min_ped_clear = 0.0
                ped_ttc_pen   = 0.0

            yaw_end = yaw + w * DT_PRED
            px_end, py_end = rollout_pose(v, w, DT_PRED)
            
            bearing_end = math.atan2(ty - py_end, tx - px_end)   
            ang_now = abs((goal_angle  - yaw     + math.pi) % (2*math.pi) - math.pi)
            ang_end = abs((bearing_end - yaw_end + math.pi) % (2*math.pi) - math.pi)
            
            ang_now_db = _deadband(ang_now, HEADING_DEADBAND_RAD)
            ang_end_db = _deadband(ang_end, HEADING_DEADBAND_RAD)

            cost_goal_dist = 0.0
            if near_goal:
                dist_end = math.hypot(tx - px_end, ty - py_end)
                cost_goal_dist = -dist_end

            carrot_pull = -math.hypot(tx - px_end, ty - py_end)

            CLEAR_SOFT_M = 0.80; W_CLEAR_SOFT = 6.0
            soft_pen = W_CLEAR_SOFT * (CLEAR_SOFT_M - min_clear_static)**2 if min_clear_static < CLEAR_SOFT_M else 0.0

            score = (
                W_GOAL * (-ang_now_db) +
                W_HEAD * (-ang_end_db) +
                W_CLEAR * (min_clear_static) +
                W_SPEED * (v) +
                W_SMOOTH * (-abs(w - last_w)) +
                (W_GOAL_DIST * cost_goal_dist if near_goal else 0.0) +
                (W_CARROT_PULL * carrot_pull) -
                soft_pen +

                W_PED_CLEAR * (min_ped_clear) - ped_ttc_pen
            )

            if score > best_score:
                best_score, best_cmd = score, (v, w)

    blocked = False
    if best_score == -float('inf'):
        scan_min_h_now = np.min(scan_LH, axis=0)
        best_idx = int(np.argmax(scan_min_h_now))
        best_angle = angles_h[best_idx]
        w_cmd = np.clip(2.2 * best_angle, -W_MAX, W_MAX)
        v_cmd = 0.0
        blocked = True
    else:
        v_cmd, w_cmd = best_cmd

        if v_cmd < 0.15 and abs(w_cmd) < 0.35 and front_min_25 > (OBSTACLE_CLEAR_MIN + 0.25):
            v_cmd = 0.18

    dbg_info = {
        "num_tracks": len(_TRACKER.tracks),
        "best_v": float(v_cmd),
        "best_w": float(w_cmd),
        "max_track_speed": float(max_track_speed),
    }

    HEADING_IN_PLACE_DEG = 40; HEADING_SLOW_DEG = 25
    heading_err = abs((goal_angle - yaw + math.pi) % (2*math.pi) - math.pi)
    if abs(w_cmd) >= TURN_IN_PLACE_W and heading_err > math.radians(HEADING_IN_PLACE_DEG):
        v_cmd = 0.0
    elif abs(w_cmd) >= TURN_SLOW_W and heading_err > math.radians(HEADING_SLOW_DEG):
        v_cmd = min(v_cmd, V_TURN_SLOW)
    else:
        KAPPA_MAX = 3.0
        v_eps = max(v_cmd, 0.05)
        if abs(w_cmd) / v_eps > KAPPA_MAX:
            w_cmd = math.copysign(KAPPA_MAX * v_eps, w_cmd)

    final_align = abs((goal_angle - (yaw + w_cmd * DT_PRED) + math.pi) % (2*math.pi) - math.pi)
    if abs(w_cmd) < W_DEADBAND and final_align < math.radians(HEADING_DEADBAND_DEG):
        w_cmd = 0.0
        
    ALIGN_FORCE_RAD = globals().get("ALIGN_FORCE_RAD", math.radians(18))
    W_MIN_TURN      = globals().get("W_MIN_TURN", 0.6)
    V_ALIGN         = globals().get("V_ALIGN", 0.10)
    
    bearing_now = math.atan2(ty - y, tx - x)
    ang_err_now = ((bearing_now - yaw + math.pi) % (2*math.pi)) - math.pi
    
    if abs(ang_err_now) > ALIGN_FORCE_RAD and abs(w_cmd) < 0.15 and front_min_25 > (FORWARD_CLEAR_DIST + 0.02):
        w_cmd = math.copysign(max(W_MIN_TURN, min(abs(ang_err_now), 1.2)), ang_err_now)
        v_cmd = min(v_cmd, V_ALIGN)

    return v_cmd, w_cmd, blocked, front_min, dbg_info