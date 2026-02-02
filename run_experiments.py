from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env, run_episode, record_video_episode
from search.random_search import RandomSearch
from search.hill_climbing import HillClimbing
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
        
        # Inline Random Search implementation to support early stopping
        rs_rng = np.random.default_rng(seed)
        rs_crash_log = []
        rs_evals_count = 0
        
        print(f"Running Random Search for {MAX_EVALS} scenarios...")
        for i in range(MAX_EVALS):
            cfg = rs.sample_random_config(rs_rng)
            episode_seed = int(rs_rng.integers(1e9))
            
            crashed, ts = run_episode(env_id, cfg, policy, defaults, episode_seed)
            rs_evals_count += 1
            
            if crashed:
                print(f"💥 Collision: scenario {i}, seed={episode_seed}")
                rs_crash_log.append({"cfg": cfg, "seed": episode_seed})
                record_video_episode(env_id, cfg, policy, defaults, episode_seed, out_dir="videos")
                break # Early stopping!
            # else:
            #     print(f"No Crash: scenario {i}, seed={episode_seed}")

        rs_runtime = time() - start_time
        rs_crashes = rs_crash_log
        
        results['random_search'].append(len(rs_crashes))
        results['rs_runtimes'].append(rs_runtime)
        # Use actual evaluations count
        results['rs_evals_to_crash'].append(rs_evals_count)
        print(f"Crashes: {len(rs_crashes)}, Runtime: {rs_runtime:.2f}s, Evals: {rs_evals_count}")
        
        print("Hill Climbing Experiment")
        start_time = time()
        hc = HillClimbing(env_id, base_cfg, param_spec, policy, defaults)
        hc_result = hc.run_search(iterations=MAX_EVALS, seed=seed)
        hc_runtime = time() - start_time
        
        hc_crashes = []
        hc_evals = MAX_EVALS
        
        if isinstance(hc_result, list):
            hc_crashes = hc_result
             # Fallback if evaluations missing (shouldn't happen with updated code)
            hc_evals = len(hc_crashes) if hc_crashes else MAX_EVALS
        elif isinstance(hc_result, dict):
            hc_crashes = hc_result.get('crashes', [])
            hc_evals = hc_result.get('evaluations', MAX_EVALS)

        results['hill_climbing'].append(len(hc_crashes))
        results['hc_runtimes'].append(hc_runtime)
        results['hc_evals_to_crash'].append(hc_evals)
        print(f"Crashes: {len(hc_crashes)}, Runtime: {hc_runtime:.2f}s, Evals: {hc_evals}")
    
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

    print(f"\Summary:")
    print(f"{'Metric':<45} {'Random Search':>15} {'Hill Climbing':>15}")
    print(f"{'Total Crashes Found':<45} {rs_total_crashes:>15} {hc_total_crashes:>15}")
    print(f"{'Avg Crashes per Seed':<45} {rs_total_crashes/N_SEEDS:>15.2f} {hc_total_crashes/N_SEEDS:>15.2f}")
    print(f"{'Avg Evaluations to Crash':<45} {rs_avg_evals:>15.1f} {hc_avg_evals:>15.1f}")
    print(f"{'Std Dev Evaluations':<45} {np.std(results['rs_evals_to_crash']):>15.1f} {np.std(results['hc_evals_to_crash']):>15.1f}")
    print(f"{'Avg Runtime (seconds)':<45} {rs_avg_runtime:>15.2f} {hc_avg_runtime:>15.2f}")
    print(f"{'Total Runtime (seconds)':<45} {np.sum(results['rs_runtimes']):>15.2f} {np.sum(results['hc_runtimes']):>15.2f}")
    print(f"{'Collision Found':<45} {str(rs_found_collision):>15} {str(hc_found_collision):>15}")
    print(f"{'Efficiency Ratio (HC/RS)':<45} {'baseline':>15} {efficiency_ratio:>15.2f}x")

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