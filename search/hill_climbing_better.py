"""
Assignment 3 — Scenario-Based Testing of an RL Agent (Hill Climbing)

You MUST implement:
    - compute_objectives_from_time_series
    - compute_fitness
    - mutate_config
    - hill_climb

DO NOT change function signatures.
You MAY add helper functions.

Goal
----
Find a scenario (environment configuration) that triggers a collision.
If you cannot trigger a collision, minimize the minimum distance between the ego
vehicle and any other vehicle across the episode.

Black-box requirement
---------------------
Your evaluation must rely only on observable behavior during execution:
- crashed flag from the environment
- time-series data returned by run_episode (positions, lane_id, etc.)
No internal policy/model details beyond calling policy(obs, info).
"""

import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from envs.highway_env_utils import run_episode


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute your objective values from the recorded time-series.

    The time_series is a list of frames. Each frame typically contains:
      - frame["crashed"]: bool
      - frame["ego"]: dict or None, e.g. {"pos":[x,y], "lane_id":..., "length":..., "width":...}
      - frame["others"]: list of dicts with positions, lane_id, etc.

    Minimum requirements (suggested):
      - crash_count: 1 if any collision happened, else 0
      - min_distance: minimum distance between ego and any other vehicle over time (float)

    Return a dictionary, e.g.:
        {
          "crash_count": 0 or 1,
          "min_distance": float
        }

    NOTE: If you want, you can add more objectives (lane-specific distances, time-to-crash, etc.)
    but keep the keys above at least.
    """
    crashed = any(f.get("crashed", False) for f in time_series)
    min_dist = float("inf")

    for frame in time_series:
        ego = frame.get("ego")
        others = frame.get("others", [])
        if ego is None:
            continue

        ex, ey = ego["pos"]
        e_hl = ego["length"] / 2
        e_hw = ego["width"] / 2

        for other in others:
            ox, oy = other["pos"]
            o_hl = other["length"] / 2
            o_hw = other["width"] / 2

            dx = abs(ex - ox) - (e_hl + o_hl)
            dy = abs(ey - oy) - (e_hw + o_hw)
            
            if dx < 0 and dy < 0:
                dist = -np.sqrt(abs(dx * dx + dy * dy))  
            else:
                dx = max(dx, 0.0)
                dy = max(dy, 0.0)
                dist = np.sqrt(dx * dx + dy * dy)

            min_dist = min(min_dist, dist)

    return {
        "crash_count": int(crashed),
        "min_distance": min_dist if min_dist != float("inf") else 100.0
    }


def compute_fitness(objectives: Dict[str, Any]) -> float:
    """
    Convert objectives into ONE scalar fitness value to MINIMIZE.

    Requirement:
    - Any crashing scenario must be strictly better than any non-crashing scenario.

    Examples:
    - If crash_count==1: fitness = -1 (best)
    - Else: fitness = min_distance (smaller is better)

    You can design a more refined scalarization if desired.
    """
    if objectives["crash_count"] > 0:
        return -1.0
    
    return objectives["min_distance"]


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Generate ONE neighbor configuration by mutating the current scenario.

    Inputs:
      - cfg: current scenario dict (e.g., vehicles_count, initial_spacing, ego_spacing, initial_lane_id)
      - param_spec: search space bounds, types (int/float), min/max
      - rng: random generator

    Requirements:
      - Do NOT modify cfg in-place (return a copy).
      - Keep mutated values within [min, max] from param_spec.
      - If you mutate lanes_count, keep initial_lane_id valid (0..lanes_count-1).

    Students can implement:
      - single-parameter mutation (recommended baseline)
      - multiple-parameter mutation
      - adaptive step sizes, etc.
    """
    mutated = copy.deepcopy(cfg)
    num_params = rng.choice([1, 2])
    param_names = list(param_spec.keys())
    chosen_params = rng.choice(param_names, size=num_params, replace=False)
    
    for param_name in chosen_params:
        spec = param_spec[param_name]
        param_type = spec["type"]
        min_val = spec["min"]
        max_val = spec["max"]
        
        current_val = mutated.get(param_name)
        
        # If parameter doesn't exist, initialize it to middle of range
        if current_val is None:
            print("MMMMM")
            if param_type == "int":
                current_val = (min_val + max_val) // 2
                print("MMMMM2")
            else:
                current_val = (min_val + max_val) / 2.0
                print("MMMMM3")
            mutated[param_name] = current_val
        
        # Mutate with 20% step size
        range_size = max_val - min_val
        step = 0.20 * range_size
        noise = rng.uniform(-step, step)
        
        if param_type == "int":
            new_val = int(round(current_val + noise))
            new_val = max(min_val, min(max_val, new_val))
        else:
            new_val = current_val + noise
            new_val = max(min_val, min(max_val, new_val))
        mutated[param_name] = new_val
    
    if "lanes_count" in mutated:
        lanes = mutated["lanes_count"]
        if "initial_lane_id" in mutated:
            mutated["initial_lane_id"] = max(0, min(lanes - 1, mutated["initial_lane_id"]))
    
    return mutated


# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

def hill_climb(
    env_id: str,
    base_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    policy,
    defaults: Dict[str, Any],
    seed: int = 0,
    iterations: int = 100,
    neighbors_per_iter: int = 10,
) -> Dict[str, Any]:
    """
    Hill climbing loop.

    You should:
      1) Start from an initial scenario (base_cfg or random sample).
      2) Evaluate it by running:
            crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
         Then compute objectives + fitness.
      3) For each iteration:
            - Generate neighbors_per_iter neighbors using mutate_config
            - Evaluate each neighbor
            - Select the best neighbor
            - Accept it if it improves fitness (or implement another acceptance rule)
            - Optionally stop early if a crash is found
      4) Return the best scenario found and enough info to reproduce.

    Return dict MUST contain at least:
        {
          "best_cfg": Dict[str, Any],
          "best_objectives": Dict[str, Any],
          "best_fitness": float,
          "best_seed_base": int,
          "history": List[float]
        }

    Optional but useful:
        - "best_time_series": ts
        - "evaluations": int
    """
    rng = np.random.default_rng(seed)
    current_cfg = dict(base_cfg)
    
    # Initialize search parameters to middle of range
    for param_name, spec in param_spec.items():
        param_type = spec["type"]
        min_val = spec["min"]
        max_val = spec["max"]
        if param_type == "int":
            current_cfg[param_name] = (min_val + max_val) // 2
        else:
            current_cfg[param_name] = (min_val + max_val) / 2.0

    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    current_fitness = compute_fitness(obj)
    
    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fitness = current_fitness
    best_seed_base = seed_base
    
    history = [best_fitness]
    evaluations = 1
    no_improvement = 0
    
    print(f"Iteration 0: fitness={best_fitness:.4f}, min_distance={best_obj['min_distance']:.4f}, crashed={best_obj['crash_count']}")
    
    for iteration in range(iterations):
        best_neighbor = None
        best_neighbor_fitness = float('inf')
        
        for _ in range(neighbors_per_iter):
            neighbor_cfg = mutate_config(current_cfg, param_spec, rng)
            neighbor_seed = int(rng.integers(1e9))
            crashed, ts = run_episode(env_id, neighbor_cfg, policy, defaults, neighbor_seed)
            neighbor_obj = compute_objectives_from_time_series(ts)
            neighbor_fitness = compute_fitness(neighbor_obj)
            evaluations += 1
            
            if neighbor_fitness < best_neighbor_fitness or \
               (neighbor_fitness == best_neighbor_fitness and best_neighbor_fitness <= 0.01):
                best_neighbor = (neighbor_cfg, neighbor_obj, neighbor_fitness, neighbor_seed)
                best_neighbor_fitness = neighbor_fitness
        
        accept = best_neighbor_fitness < current_fitness
        if best_neighbor_fitness == current_fitness and current_fitness <= 0.01:
            accept = rng.random() < 0.3  
        
        if accept:
            current_cfg, obj, current_fitness, seed_base = best_neighbor
            
            if current_fitness < best_fitness:
                best_cfg = copy.deepcopy(current_cfg)
                best_obj = dict(obj)
                best_fitness = current_fitness
                best_seed_base = seed_base
                no_improvement = 0
            else:
                no_improvement += 1
        else:
            no_improvement += 1
        
        history.append(best_fitness)
        print(f"Iteration {iteration+1}: fitness={best_fitness:.4f}, min_distance={best_obj['min_distance']:.4f}, crashed={best_obj['crash_count']}")
        
        if best_obj["crash_count"] > 0:
            break
        
        restart_threshold = 3 if best_fitness <= 0.01 else 5
        if no_improvement > restart_threshold:
            print("Restarting")
            # Generate random aggressive config
            for param_name, spec in param_spec.items():
                param_type = spec["type"]
                min_val = spec["min"]
                max_val = spec["max"]
                if param_type == "int":
                    current_cfg[param_name] = rng.integers(min_val, max_val + 1)
                else:
                    current_cfg[param_name] = rng.uniform(min_val, max_val)
            
            seed_base = int(rng.integers(1e9))
            crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
            obj = compute_objectives_from_time_series(ts)
            current_fitness = compute_fitness(obj)
            evaluations += 1
            no_improvement = 0
    
    neighbor_obj = compute_objectives_from_time_series(ts)

    if neighbor_obj["crash_count"] > 0:
      return {
        "best_cfg": neighbor_cfg,
        "best_objectives": neighbor_obj,
        "best_fitness": -1.0,
        "best_seed_base": neighbor_seed,
        "history": history,
        "evaluations": evaluations,
        "best_time_series": ts
    }

