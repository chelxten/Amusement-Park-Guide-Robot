import numpy as np
import math

DEFAULT_MAP_SIZE = 50.0  
MAP_SIZE = 50.0
RESOLUTION = 0.025
GRID_SIZE = int(MAP_SIZE / RESOLUTION)
LO_UNKNOWN = 0.0
class DynamicSLAM:

    def __init__(self, map_size_m: float = DEFAULT_MAP_SIZE):
        self.map_size_m: float = map_size_m
        self.resolution: float | None = None
        self.grid_size: int | None = None

        self.static_map: np.ndarray | None = None

        self.dynamic_prob: np.ndarray | None = None

        self.decay_base = 0.01         
        self.hit_increment = 0.15     
        self.confidence_decay = 0.005  
        self.occ_thresh = 0.5     

        self.angle_offset = -math.pi/2 + 1.560

    #  Map 
    def set_static_map_from_file(self, filename: str):
        
        occ = np.load(filename)
        if occ.ndim != 2 or occ.shape[0] != occ.shape[1]:
            raise ValueError(f"Static map must be square 2D array, got shape {occ.shape}")

        N = occ.shape[0]
        self.grid_size = N
        self.resolution = float(self.map_size_m) / float(N)

        static_bin = np.zeros_like(occ, dtype=np.uint8)
        static_bin[occ > 0.6] = 1
        static_bin[occ < 0.4] = 0
        self.static_map = static_bin

        self.dynamic_prob = np.zeros((N, N), dtype=np.float32)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        if self.resolution is None:
            raise RuntimeError("Static map not set yet; call set_static_map_from_file first.")
        gx = int((x + self.map_size_m / 2.0) / self.resolution)
        gy = int((y + self.map_size_m / 2.0) / self.resolution)
        return gx, gy

    def update_from_scan(self, pose, scan, num_rays, fov, max_range):

        if self.dynamic_prob is None:
            raise RuntimeError("Dynamic layer not initialized (load static map first).")

        x, y, yaw = pose
        observed = np.zeros_like(self.dynamic_prob, dtype=bool)

        for i in range(num_rays):
            r = float(scan[i])
            if not np.isfinite(r) or r <= 0.02 or r > max_range:
                continue

            a = (i / (num_rays - 1)) * fov - fov / 2.0
            a += self.angle_offset
            wx = x + r * math.cos(yaw + a)
            wy = y + r * math.sin(yaw + a)

            gx, gy = self.world_to_grid(wx, wy)
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                self.dynamic_prob[gx, gy] += self.hit_increment
                observed[gx, gy] = True

        np.clip(self.dynamic_prob, 0.0, 1.0, out=self.dynamic_prob)

        decay_map = self.decay_base + self.confidence_decay * self.dynamic_prob
        self.dynamic_prob[~observed] -= decay_map[~observed]
        np.clip(self.dynamic_prob, 0.0, 1.0, out=self.dynamic_prob)

    def get_combined_map(self) -> np.ndarray:

        if self.static_map is None or self.dynamic_prob is None:
            raise RuntimeError("Maps not initialized.")
        dynamic_obs = (self.dynamic_prob > self.occ_thresh).astype(np.uint8)
        return np.maximum(self.static_map, dynamic_obs)

    def new_obstacle_in_path(self, path_grid_indices) -> tuple[bool, tuple[int, int] | None]:

        if self.dynamic_prob is None:
            return False, None
        for gx, gy in path_grid_indices:
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                if self.dynamic_prob[gx, gy] > self.occ_thresh:
                    return True, (gx, gy)
        return False, None