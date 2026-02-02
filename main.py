from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch
from search.hill_climbing import HillClimbing

def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    # search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    # crashes = search.run_search(n_scenarios=50, seed=11)
    search = HillClimbing(env_id, base_cfg, param_spec, policy, defaults)
    crashes = search.run_search(seed=42, neighbors_per_iter=5)

    print(f"✅ Found {len(crashes)} crashes.")
    #if crashes:
    #    print(crashes)

if __name__ == "__main__":
    main()