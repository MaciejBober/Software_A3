from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch
from search.hill_climbing_better import hill_climb


def main():
    """Run experiments comparing Random Search and Hill Climbing."""
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)
    
    N_SEEDS = 10
    MAX_EVALS = 200
    SEEDS = list(range(42, 42 + N_SEEDS))
    
    results = {'random_search': [], 'hill_climbing': []}
    
    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print('='*70)
        
        # # Random Search
        # print("  Random Search...")
        # rs = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
        # rs_crashes = rs.run_search(n_scenarios=MAX_EVALS, seed=seed)
        # results['random_search'].append(len(rs_crashes))
        # print(f"    Crashes: {len(rs_crashes)}")
        
        # Hill Climbing
        print("  Hill Climbing...")
        hc = hill_climb(env_id, base_cfg, param_spec, policy, defaults)
        hc_crashes = hc.run_search(n_scenarios=MAX_EVALS, seed=seed)
        results['hill_climbing'].append(len(hc_crashes))
        print(f"    Crashes: {len(hc_crashes)}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Seeds: {N_SEEDS}, Max Evals: {MAX_EVALS}")
    
    # rs_total = sum(results['random_search'])
    hc_total = sum(results['hill_climbing'])
    
    # print(f"\nRandom Search:  {rs_total} total crashes ({rs_total/N_SEEDS:.1f} avg)")
    print(f"Hill Climbing:  {hc_total} total crashes ({hc_total/N_SEEDS:.1f} avg)")
    print('='*70)


if __name__ == "__main__":
    main()
