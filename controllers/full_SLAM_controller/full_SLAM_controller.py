from controller import Robot, Keyboard
import numpy as np
import atexit
from math import cos, sin
import os

MAP_SIZE = 50.0          
RESOLUTION = 0.025
GRID_SIZE = int(MAP_SIZE / RESOLUTION)
TIME_STEP = 32

# LiDAR ranges
MAX_RANGE = 35.0           
NOHIT_MARGIN_M = 0.50       
END_NEAR_MARGIN_M = 0.35    

# Update speed
LIVE_PLOT = False
DECIMATE_RAYS = 2
MAP_EVERY_N = 2
SCANMATCH_EVERY = 5
CLIP_EVERY = 20
PATH_EVERY = 3
RAY_SKIP_CELLS = 1
LIDAR_PERIOD_MS = 64
_SAVED_ONCE = False

OCCUPIED_NEAR_IGNORE = 0.3  
FREE_END_MARGIN_M     = 0.08 
MEDIAN_WIN            = 5    

# Occupancy log-odds 
LO_UNKNOWN = 0.0
LO_OCC = 1.5
LO_FREE = -0.8
LO_MIN = -5
LO_MAX = 5

# Occupied padding 
PAD_NORMAL_WIDTH_M = 0.09    
MIN_OCC_DIST_CELLS = 3       

# LiDAR yaw offset 
ANGLE_OFFSET = -np.pi/2 + 1.560

log_odds_grid = LO_UNKNOWN * np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
robot_path = []

# Build padding offsets 
_PAD_RADIUS_CELLS = max(1, int(round(PAD_NORMAL_WIDTH_M / RESOLUTION)))
PAD_OFFSETS = []
for dx in range(-_PAD_RADIUS_CELLS, _PAD_RADIUS_CELLS + 1):
    for dy in range(-_PAD_RADIUS_CELLS, _PAD_RADIUS_CELLS + 1):
        if dx*dx + dy*dy <= _PAD_RADIUS_CELLS*_PAD_RADIUS_CELLS:
            PAD_OFFSETS.append((dx, dy))

def world_to_grid(x, y):
    gx = int((x + MAP_SIZE / 2) / RESOLUTION)
    gy = int((y + MAP_SIZE / 2) / RESOLUTION)
    return gx, gy

def logodds_to_prob(l):
    return 1 - 1 / (1 + np.exp(l))

def bresenham(x0, y0, x1, y1):
    pts = []
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2
        while x != x1:
            pts.append((x, y))
            err -= dy
            if err < 0:
                y += sy; err += dx
            x += sx
    else:
        err = dy / 2
        while y != y1:
            pts.append((x, y))
            err -= dx
            if err < 0:
                x += sx; err += dy
            y += sy
    pts.append((x1, y1))
    return pts

def smooth_map(grid):
    import scipy.ndimage
    return scipy.ndimage.median_filter(grid, size=2)

def scan_match(x, y, yaw, scan, log_odds_grid, num_rays, fov, max_range_used):

    best_score = -np.inf
    best_pose = (x, y, yaw)
    deltas = np.linspace(-0.08, 0.08, 5)
    angles = np.deg2rad(np.linspace(-4, 4, 5))
    for dx in deltas:
        for dy in deltas:
            for dth in angles:
                score = 0.0
                for i in range(0, num_rays, DECIMATE_RAYS):
                    d = scan[i]
                    if not np.isfinite(d) or d <= 0.05 or d > max_range_used:
                        continue
                    a = (i / (num_rays - 1)) * fov - fov / 2
                    a += ANGLE_OFFSET
                    wx = (x + dx) + d * cos((yaw + dth) + a)
                    wy = (y + dy) + d * sin((yaw + dth) + a)
                    gx, gy = world_to_grid(wx, wy)
                    if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                        score += log_odds_grid[gx, gy]
                if score > best_score:
                    best_score = score
                    best_pose = (x + dx, y + dy, yaw + dth)
    return best_pose

def main():
    robot = Robot()
    keyboard = Keyboard(); keyboard.enable(TIME_STEP)

    gps = robot.getDevice("gps"); gps.enable(TIME_STEP)
    imu = robot.getDevice("inertial unit"); imu.enable(TIME_STEP)
    lidar = robot.getDevice("lidar"); lidar.enable(LIDAR_PERIOD_MS)

    num_rays = lidar.getHorizontalResolution()
    fov = lidar.getFov()
    phys_max = float(lidar.getMaxRange())             
    max_range_used = min(phys_max, MAX_RANGE)         
    occ_far_cut = max_range_used - END_NEAR_MARGIN_M  

    left_motor  = robot.getDevice("wheel_left_joint")
    right_motor = robot.getDevice("wheel_right_joint")
    left_motor.setPosition(float('inf')); right_motor.setPosition(float('inf'))
    max_speed = 3.0

    step = 0
    last_xcorr = last_ycorr = last_yawc = 0.0

    def save_path():
        global _SAVED_ONCE
        if _SAVED_ONCE: 
            return
        _SAVED_ONCE = True
    
        import time, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.ndimage import median_filter, binary_opening, generate_binary_structure
    
        outdir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(outdir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
    
        occ_only_png = os.path.join(outdir, f"occupancy_grid_only_{ts}.png")
        occ_with_path_png = os.path.join(outdir, f"occupancy_grid_with_path_{ts}.png")
        npy_path = os.path.join(outdir, f"occupancy_grid_{ts}.npy")
    
        occ = logodds_to_prob(log_odds_grid)
        occ = median_filter(occ, size=2)
    
        try:
            occ_mask = occ > 0.55
            se = generate_binary_structure(2, 1)
            occ_mask_clean = binary_opening(occ_mask, structure=se, iterations=1)
            occ = np.where(occ_mask & ~occ_mask_clean, 0.5, occ)
        except Exception:
            pass
    
        # 1- Occupancy-only image 
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.imshow(
            occ.T, cmap='gray_r', vmin=0, vmax=1, origin='lower',
            extent=[-MAP_SIZE/2, MAP_SIZE/2, -MAP_SIZE/2, MAP_SIZE/2]
        )
        ax1.set_xlim(-MAP_SIZE/2, MAP_SIZE/2); ax1.set_ylim(-MAP_SIZE/2, MAP_SIZE/2)
        ax1.axis('equal'); ax1.axis('off')
        fig1.savefig(occ_only_png, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig1)
    
        # 2- Occupancy + robot path overlay 
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.imshow(
            occ.T, cmap='gray_r', vmin=0, vmax=1, origin='lower',
            extent=[-MAP_SIZE/2, MAP_SIZE/2, -MAP_SIZE/2, MAP_SIZE/2]
        )
        if robot_path:
            xs, ys = zip(*robot_path)
            ax2.plot(xs, ys, 'r-', linewidth=2)
            ax2.plot(xs[0], ys[0], 'go', markersize=8)   
            ax2.plot(xs[-1], ys[-1], 'ro', markersize=8) 
        ax2.set_xlim(-MAP_SIZE/2, MAP_SIZE/2); ax2.set_ylim(-MAP_SIZE/2, MAP_SIZE/2)
        ax2.axis('equal'); ax2.axis('off')
        fig2.savefig(occ_with_path_png, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig2)
    

        np.save(npy_path, occ)
    
        print(" Saved:")
        print(" •", os.path.abspath(occ_only_png))
        print(" •", os.path.abspath(occ_with_path_png))
        print(" •", os.path.abspath(npy_path))
        
    atexit.register(save_path)

    try:
        while robot.step(TIME_STEP) != -1:
            step += 1

            key = keyboard.getKey()
            if key in (ord('Q'), ord('q')): break
            l = r = 0.0
            if key == Keyboard.UP:    l = r = max_speed
            if key == Keyboard.DOWN:  l = r = -max_speed
            if key == Keyboard.LEFT:  l = -0.5*max_speed; r = 0.5*max_speed
            if key == Keyboard.RIGHT: l =  0.5*max_speed; r = -0.5*max_speed
            left_motor.setVelocity(l); right_motor.setVelocity(r)

            pos = gps.getValues()
            x = pos[0]; y = pos[1]
            yaw = imu.getRollPitchYaw()[2]
            scan = lidar.getRangeImage()

            if step % SCANMATCH_EVERY == 0:
                x_corr, y_corr, yaw_corr = scan_match(
                    x, y, yaw, scan, log_odds_grid, num_rays, fov, max_range_used
                )
                last_xcorr, last_ycorr, last_yawc = x_corr, y_corr, yaw_corr
            else:
                x_corr, y_corr, yaw_corr = last_xcorr, last_ycorr, last_yawc

            if step % PATH_EVERY == 0:
                if (not robot_path or
                    abs(x_corr - robot_path[-1][0]) > 0.01 or
                    abs(y_corr - robot_path[-1][1]) > 0.01):
                    robot_path.append((x_corr, y_corr))

            if step % MAP_EVERY_N == 0:
                gx0, gy0 = world_to_grid(x_corr, y_corr)

                scan_arr = np.asarray(scan, dtype=np.float32)
                scan_arr[~np.isfinite(scan_arr)] = np.inf
                near_phys_cut = max_range_used - NOHIT_MARGIN_M
                scan_arr[scan_arr >= near_phys_cut] = np.inf
                if MEDIAN_WIN > 1:
                    from scipy.ndimage import median_filter as _medf
                    scan_f = _medf(scan_arr, size=MEDIAN_WIN, mode='nearest')
                else:
                    scan_f = scan_arr

                for i in range(0, num_rays, DECIMATE_RAYS):
                    d_raw = float(scan_f[i])

                    no_hit = (not np.isfinite(d_raw)) or (d_raw >= near_phys_cut)
                    if (not no_hit) and (d_raw <= 0.02 or d_raw > max_range_used):
                        continue  
                        

                    a = (i / (num_rays - 1)) * fov - fov / 2
                    a += ANGLE_OFFSET

                    if no_hit:
                       
                        d_free = max(0.0, max_range_used - FREE_END_MARGIN_M)
                        wx_end = x_corr + d_free * cos(yaw_corr + a)
                        wy_end = y_corr + d_free * sin(yaw_corr + a)
                        gx1, gy1 = world_to_grid(wx_end, wy_end)

                        if (0 <= gx0 < GRID_SIZE and 0 <= gy0 < GRID_SIZE and
                            0 <= gx1 < GRID_SIZE and 0 <= gy1 < GRID_SIZE):
                            pts = bresenham(gx0, gy0, gx1, gy1)
                            if RAY_SKIP_CELLS > 1 and len(pts) > 1:
                                pts = pts[::RAY_SKIP_CELLS]
                            for px, py in pts:
                                val = log_odds_grid[px, py] + LO_FREE
                                log_odds_grid[px, py] = min(LO_MAX, max(LO_MIN, val))
                        continue  

                    d_free = max(0.0, d_raw - FREE_END_MARGIN_M)
                    wx_free = x_corr + d_free * cos(yaw_corr + a)
                    wy_free = y_corr + d_free * sin(yaw_corr + a)
                    gxF, gyF = world_to_grid(wx_free, wy_free)

                    if (0 <= gx0 < GRID_SIZE and 0 <= gy0 < GRID_SIZE and
                        0 <= gxF < GRID_SIZE and 0 <= gyF < GRID_SIZE):
                        pts = bresenham(gx0, gy0, gxF, gyF)
                        if RAY_SKIP_CELLS > 1 and len(pts) > 1:
                            pts = pts[::RAY_SKIP_CELLS]
                        for px, py in pts:
                            val = log_odds_grid[px, py] + LO_FREE
                            log_odds_grid[px, py] = min(LO_MAX, max(LO_MIN, val))

                    wx_hit = x_corr + d_raw * cos(yaw_corr + a)
                    wy_hit = y_corr + d_raw * sin(yaw_corr + a)
                    gx1, gy1 = world_to_grid(wx_hit, wy_hit)

                    far_enough_from_robot = (
                        abs(gx1 - gx0) + abs(gy1 - gy0) >= MIN_OCC_DIST_CELLS
                    )
                    within_far_cut = d_raw <= occ_far_cut

                    if (d_raw >= OCCUPIED_NEAR_IGNORE) and within_far_cut and far_enough_from_robot:
                        if 0 <= gx1 < GRID_SIZE and 0 <= gy1 < GRID_SIZE:
                            for dx_b, dy_b in PAD_OFFSETS:
                                bx = gx1 + dx_b; by = gy1 + dy_b
                                if 0 <= bx < GRID_SIZE and 0 <= by < GRID_SIZE:
                                    val = log_odds_grid[bx, by] + LO_OCC
                                    log_odds_grid[bx, by] = min(LO_MAX, max(LO_MIN, val))

            if step % CLIP_EVERY == 0:
                np.clip(log_odds_grid, LO_MIN, LO_MAX, out=log_odds_grid)

    finally:
        try:
            save_path()
        except Exception as e:
            print(f"⚠️ save_path() failed: {e}")

if __name__ == "__main__":
    if LIVE_PLOT:
        pass
    main()