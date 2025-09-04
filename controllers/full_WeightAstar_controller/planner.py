import heapq, numpy as np, time
from typing import List, Tuple, Dict, Any

def weighted_a_star(
    start: Tuple[int,int],
    goal: Tuple[int,int],
    grid: np.ndarray,
    penalty: np.ndarray | None = None,
    w: float = 1.5,
    *,
    cell_size_m: float = 1.0,
    is_replan: bool = False,
    time_budget_ms: float | None = None
) -> tuple[list[tuple[int,int]], Dict[str, Any]]:

    def h(a, b) -> float:
        return np.hypot(a[0] - b[0], a[1] - b[1])

    t0 = time.perf_counter()

    open_set: list[tuple[float, tuple[int,int]]] = []
    heapq.heappush(open_set, (0.0, start))
    came_from: dict[tuple[int,int], tuple[int,int]] = {}
    g: dict[tuple[int,int], float] = {start: 0.0}

    gx, gy = goal
    H, W = grid.shape

    nodes_expanded = 0          
    nodes_generated = 0       
    open_list_max = 1
    ok = False
    fail_reason = None

    if not (0 <= gx < H and 0 <= gy < W):
        fail_reason = "goal_out_of_bounds"
    elif grid[gx, gy] > 0:
        fail_reason = "goal_in_collision"

    if fail_reason:
        t_ms = (time.perf_counter() - t0) * 1000.0
        return [], {
            "ok": 0,
            "fail_reason": fail_reason,
            "plan_time_ms": t_ms,
            "is_replan": int(is_replan),
            "nodes_expanded": nodes_expanded,
            "nodes_generated": nodes_generated,
            "open_list_max": open_list_max,
            "path_cost_grid": None,
            "path_len_cells": 0.0,
            "path_len_m": 0.0,
            "w": w,
        }

    budget_s = None if time_budget_ms is None else (time_budget_ms / 1000.0)

    while open_set:
        if budget_s is not None and (time.perf_counter() - t0) > budget_s:
            fail_reason = "time_budget_exceeded"
            break

        _, cur = heapq.heappop(open_set)
        nodes_expanded += 1

        if cur == (gx, gy):
            path: list[tuple[int,int]] = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            ok = True
            break

        cx, cy = cur
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(1,1),(-1,1),(1,-1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            if grid[nx, ny] > 0:
                continue
            if dx and dy:
                if grid[cx + dx, cy] > 0 or grid[cx, cy + dy] > 0:
                    continue

            step = np.hypot(dx, dy)
            p = 0.0 if penalty is None else float(penalty[nx, ny])
            tg = g[cur] + step + p
            nb = (nx, ny)

            if nb in g and tg >= g[nb]:
                continue

            g[nb] = tg
            f = tg + w * h(nb, (gx, gy))
            heapq.heappush(open_set, (f, nb))
            came_from[nb] = cur
            nodes_generated += 1
            if len(open_set) > open_list_max:
                open_list_max = len(open_set)

    t_ms = (time.perf_counter() - t0) * 1000.0

    if not ok:
        if fail_reason is None:
            fail_reason = "no_path"
        return [], {
            "ok": 0,
            "fail_reason": fail_reason,
            "plan_time_ms": t_ms,
            "is_replan": int(is_replan),
            "nodes_expanded": nodes_expanded,
            "nodes_generated": nodes_generated,
            "open_list_max": open_list_max,
            "path_cost_grid": None,
            "path_len_cells": 0.0,
            "path_len_m": 0.0,
            "w": w,
        }

    def polyline_length(points: List[tuple[int,int]]) -> float:
        if len(points) < 2:
            return 0.0
        pts = np.array(points, dtype=float)
        diffs = np.diff(pts, axis=0)
        segs = np.hypot(diffs[:,0], diffs[:,1])
        return float(segs.sum())

    path_len_cells = polyline_length(path)
    path_len_m = path_len_cells * float(cell_size_m)
    path_cost_grid = g[(gx, gy)]  

    metrics = {
        "ok": 1,
        "fail_reason": None,
        "plan_time_ms": t_ms,
        "is_replan": int(is_replan),
        "nodes_expanded": nodes_expanded,
        "nodes_generated": nodes_generated,
        "open_list_max": open_list_max,
        "path_cost_grid": path_cost_grid,
        "path_len_cells": path_len_cells,
        "path_len_m": path_len_m,
        "w": w,
    }
    return path, metrics