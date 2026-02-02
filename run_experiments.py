from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch
from search.hill_climbing import hill_climb
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time

def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)
    
    N_SEEDS = 10
    MAX_EVALS = 200
    SEEDS = list(range(42, 42 + N_SEEDS))
    
    results = {
        'random_search': [],
        'hill_climbing': [],
        'rs_evals_to_crash': [],
        'hc_evals_to_crash': [],
        'rs_min_distances': [],
        'hc_min_distances': [],
        'rs_runtimes': [],
        'hc_runtimes': []
    }
    
    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print('='*70)
        
        print("Random Search Experiment")
        start_time = time()
        rs = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
        rs_result = rs.run_search(n_scenarios=MAX_EVALS, seed=seed)
        rs_runtime = time() - start_time
        rs_crashes = rs_result if isinstance(rs_result, list) else rs_result.get('crashes', [])
        results['random_search'].append(len(rs_crashes))
        results['rs_runtimes'].append(rs_runtime)
        results['rs_evals_to_crash'].append(len(rs_crashes) if rs_crashes else MAX_EVALS)
        print(f"Crashes: {len(rs_crashes)}, Runtime: {rs_runtime:.2f}s")
        
        print("Hill Climbing Experiment")
        start_time = time()
        hc = hill_climb(env_id, base_cfg, param_spec, policy, defaults)
        hc_result = hc.run_search(n_scenarios=MAX_EVALS, seed=seed)
        hc_runtime = time() - start_time
        hc_crashes = hc_result if isinstance(hc_result, list) else hc_result.get('crashes', [])
        results['hill_climbing'].append(len(hc_crashes))
        results['hc_runtimes'].append(hc_runtime)
        results['hc_evals_to_crash'].append(len(hc_crashes) if hc_crashes else MAX_EVALS)
        print(f"Crashes: {len(hc_crashes)}, Runtime: {hc_runtime:.2f}s")
    
    print(f"Configuration: Seeds: {N_SEEDS}, Max Evaluations: {MAX_EVALS}")
    
    rs_found_collision = sum(results['random_search']) > 0
    hc_found_collision = sum(results['hill_climbing']) > 0
    
    print(f"Random Search found collision: {rs_found_collision}")
    print(f"Hill Climber found collision: {hc_found_collision}")
    
    rs_total_crashes = sum(results['random_search'])
    hc_total_crashes = sum(results['hill_climbing'])
    print(f"Random Search: {rs_total_crashes} total crashes across all seeds")
    print(f"Hill Climbing: {hc_total_crashes} total crashes across all seeds")
    print(f"Random Search: {rs_total_crashes/N_SEEDS:.1f} avg crashes per seed")
    print(f"Hill Climbing: {hc_total_crashes/N_SEEDS:.1f} avg crashes per seed")
    
    rs_avg_evals = np.mean(results['rs_evals_to_crash'])
    hc_avg_evals = np.mean(results['hc_evals_to_crash'])
    rs_avg_runtime = np.mean(results['rs_runtimes'])
    hc_avg_runtime = np.mean(results['hc_runtimes'])
    
    print(f"Random Search: {rs_avg_evals:.1f} avg ± {np.std(results['rs_evals_to_crash']):.1f}")
    print(f"Hill Climber:  {hc_avg_evals:.1f} avg ± {np.std(results['hc_evals_to_crash']):.1f}")
    
    print(f"\nRuntime (seconds):")
    print(f"Random Search: {rs_avg_runtime:.2f}s avg ({np.sum(results['rs_runtimes']):.2f}s total)")
    print(f"Hill Climber:  {hc_avg_runtime:.2f}s avg ({np.sum(results['hc_runtimes']):.2f}s total)")
    
    efficiency_ratio = hc_avg_evals / rs_avg_evals if rs_avg_evals > 0 else 1
    print(f"\nEfficiency Ratio (HC/RS evals): {efficiency_ratio:.2f}x")

    print(f"\nScalability:")
    print(f"Max evaluations used: {MAX_EVALS}")
    print(f"Random Search: Scales linearly with evaluations, O(n)")
    print(f"Hill Climber: Scales with iterations and neighbors, O(i*n) where i=iterations, n=neighbors_per_iter")
    
    methods = ["Random Search", "Hill Climbing"]
    avg_evals = [rs_avg_evals, hc_avg_evals]
    crash_counts = [rs_total_crashes, hc_total_crashes]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(methods, avg_evals, alpha=0.8, color=['#1f77b4', '#ff7f0e'])
    axes[0].set_title("Average Evaluations to First Crash")
    axes[0].set_ylabel("Evaluations")
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(methods, crash_counts, alpha=0.8, color=['#1f77b4', '#ff7f0e'])
    axes[1].set_title("Total Crashes")
    axes[1].set_ylabel("Crash Count")
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/comparison_plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()