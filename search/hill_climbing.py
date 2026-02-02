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
    crash_count = 0
    min_distance = float('inf')
    
    for item in time_series: 
        if item.get("crashed", False) == True: 
            crash_count += 1 
            min_distance = 0 
            return {
                "crash_count": crash_count,
                "min_distance": min_distance
            }

    
    for frame in time_series:
        ego = frame.get("ego")
        others = frame.get("others", [])
        
        if ego is None or not others:
            continue
        
        ego_pos = ego.get("pos")
        if ego_pos is None:
            continue
        
        for other in others:
            other_pos = other.get("pos")
            if other_pos is None:
                continue
            
            dx = abs(ego_pos[0] - other_pos[0])
            dy = abs(ego_pos[1] - other_pos[1])

            L = (ego["length"] / 2) + (other["length"] / 2)
            W = (ego["width"] / 2) + (other["width"] / 2)

            dx -= L
            dy -= W

            distance = max(dx, dy)
            min_distance = min(min_distance, distance)
    
    if min_distance == float('inf'):
        min_distance = 1000.0
    
    return {
        "crash_count": crash_count,
        "min_distance": min_distance
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
    crash_count = objectives.get("crash_count", 0.0)
    min_distance = objectives.get("min_distance", 1000.0)
    
    # Crashes are always better (lower fitness) than non-crashes
    if crash_count >= 1:
        return -1.0  # Best possible fitness
    else:
        # For non-crashes, minimize distance (smaller distance = better fitness)
        return min_distance


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
    mutable_params = list(param_spec.keys())
    
    param_to_mutate = rng.choice(mutable_params)
    spec = param_spec[param_to_mutate]
    
    if spec["type"] == "int":
        current_val = mutated.get(param_to_mutate, spec["min"])
        range_size = spec["max"] - spec["min"]
        step_size = range_size
        delta = rng.uniform(-step_size, step_size)
        new_val = int(np.clip(current_val + delta, spec["min"], spec["max"]))

        mutated[param_to_mutate] = new_val
        if param_to_mutate == "lanes_count":
            current_lane = mutated.get("initial_lane_id", 0)
            mutated["initial_lane_id"] = int(np.clip(current_lane, 0, new_val - 1))
    
    elif spec["type"] == "float":
        current_val = mutated.get(param_to_mutate, spec["min"])
        range_size = spec["max"] - spec["min"]
        std_dev = range_size * 0.3
        delta = rng.uniform(-std_dev, std_dev)
        new_val = float(np.clip(current_val + delta, spec["min"], spec["max"]))
        mutated[param_to_mutate] = new_val
    
    if param_to_mutate == "initial_lane_id":
        lanes = mutated.get("lanes_count", 3)
        mutated["initial_lane_id"] = int(np.clip(mutated["initial_lane_id"], 0, lanes - 1))
    
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
    
    for param, spec in param_spec.items():
        if param not in current_cfg:
            if spec["type"] == "int":
                current_cfg[param] = (spec["min"] + spec["max"]) // 2
            else:
                current_cfg[param] = (spec["min"] + spec["max"]) / 2.0
    
    lanes = current_cfg.get("lanes_count", 3)
    if "initial_lane_id" in current_cfg:
        current_cfg["initial_lane_id"] = min(current_cfg["initial_lane_id"], lanes - 1)

    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)

    if crashed: 
        print("crashed but not reported")

    obj = compute_objectives_from_time_series(ts)
    cur_fit = compute_fitness(obj)

    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base
    best_ts = ts

    history = [best_fit]
    evaluations = 1
    reset_count = 0 

    # Hill climbing main loop
    for iteration in range(iterations):
        # Early stopping if we found a crash
        if best_obj.get("crash_count", 0) >= 1:
            print(f"Crash found at iteration {iteration}!")
            break
        
        neighbors = []
        for _ in range(neighbors_per_iter):
            neighbor_cfg = mutate_config(current_cfg, param_spec, rng)

            for tt in range(5):
                neighbor_cfg = mutate_config(neighbor_cfg, param_spec, rng) 

            neighbor_seed = int(rng.integers(1e9))

            n_crashed, n_ts = run_episode(env_id, neighbor_cfg, policy, defaults, neighbor_seed)
            if n_crashed: 
                n_ts.append({"crashed": True})
                # print("neighbour crashed") 
            n_obj = compute_objectives_from_time_series(n_ts)
            n_fit = compute_fitness(n_obj)
            # print(f"n_fit = {n_fit}, n_obj = {n_obj}")
            evaluations += 1

            neighbors.append({
                "cfg": neighbor_cfg,
                "obj": n_obj,
                "fit": n_fit,
                "seed": neighbor_seed,
                "ts": n_ts
            })
        
        best_neighbor = min(neighbors, key=lambda x: x["fit"])
        
        if best_neighbor["fit"] < cur_fit:
            current_cfg = best_neighbor["cfg"]
            cur_fit = best_neighbor["fit"]
            reset_count = 0 
            
            if cur_fit < best_fit:
                best_cfg = copy.deepcopy(current_cfg)
                best_obj = dict(best_neighbor["obj"])
                best_fit = float(cur_fit)
                best_seed_base = best_neighbor["seed"]
                best_ts = best_neighbor["ts"]
                print(f"Iteration {iteration}: New best fitness = {best_fit:.4f}")

        else: 
            reset_count += 1 
            if reset_count == 5:
                cur_fit = 1000
                reset_count = 0 

        
        history.append(best_fit)
    
    return {
        "best_cfg": best_cfg,
        "best_objectives": best_obj,
        "best_fitness": best_fit,
        "best_seed_base": best_seed_base,
        "best_time_series": best_ts,
        "history": history,
        "evaluations": evaluations
    }


# ============================================================
# 4) HILL CLIMBING CLASS WRAPPER
# ============================================================

class HillClimbing:
    """
    Hill Climbing search wrapper class.
    Provides a consistent interface matching RandomSearch.
    """
    def __init__(self, env_id, base_cfg, param_spec, policy, defaults):
        self.env_id = env_id
        self.base_cfg = base_cfg
        self.param_spec = param_spec
        self.policy = policy
        self.defaults = defaults

    def run_search(self, iterations=100, neighbors_per_iter=10, seed=42, record_crashes=True):
        """
        Run hill climbing search.
        
        Args:
            iterations: Number of hill climbing iterations
            neighbors_per_iter: Number of neighbors to generate per iteration
            seed: Random seed for reproducibility
            record_crashes: Whether to record videos of crashes
        
        Returns:
            List of crash scenarios (for compatibility with RandomSearch)
        """
        print(f"Running Hill Climbing for {iterations} iterations with {neighbors_per_iter} neighbors per iteration...")
        
        result = hill_climb(
            self.env_id,
            self.base_cfg,
            self.param_spec,
            self.policy,
            self.defaults,
            seed=seed,
            iterations=iterations,
            neighbors_per_iter=neighbors_per_iter
        )
        
        print(f"\n{'='*60}")
        print(f"Hill Climbing Results:")
        print(f"{'='*60}")
        print(f"Best fitness: {result['best_fitness']:.4f}")
        print(f"Crash found: {result['best_objectives']['crash_count'] == 1}")
        print(f"Min distance: {result['best_objectives']['min_distance']:.4f}")
        print(f"Total evaluations: {result['evaluations']}")
        print(f"Best configuration: {result['best_cfg']}")
        print(f"{'='*60}\n")
        
        crash_log = []
        if result['best_objectives']['crash_count'] == 1:
            print(f"Collision found! Recording video...")
            crash_log.append({
                "cfg": result['best_cfg'],
                "seed": result['best_seed_base']
            })
            
            if record_crashes:
                from envs.highway_env_utils import record_video_episode
                record_video_episode(
                    self.env_id,
                    result['best_cfg'],
                    self.policy,
                    self.defaults,
                    result['best_seed_base'],
                    out_dir="videos"
                )
        
        return crash_log