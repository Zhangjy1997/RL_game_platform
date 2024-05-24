from onpolicy.envs.uav.environment import MultiAgentUAVEnv
from onpolicy.envs.uav.scenarios import load
import argparse
import time
import numpy as np

def UAVEnv(args):
    '''
    Creates a MultiAgentUAVEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    valid_scenario_names = ['N_v_Default1', 'N_v_interaction', 'N_v_Simple', 'N_v_Symmetry']
    # load simple_uav scenario from script
    assert args.scenario_name in valid_scenario_names, 'wrong environment name specified!'
    if args.scenario_name == 'N_v_Default1':
        scenario = load(args.scenario_name + ".py").NvDefault1(args.n_rollout_threads, args.n_rollout_threads, args.use_render)
    elif args.scenario_name == 'N_v_interaction':
        scenario = load(args.scenario_name + ".py").NvInteraction(args.n_rollout_threads, args.n_rollout_threads, args.use_render, args.team_name, args.oppo_name)
    elif args.scenario_name == 'N_v_Simple':
        scenario = load(args.scenario_name + ".py").NvSimple(args.n_rollout_threads, args.team_name, args.oppo_name)
    else:
        scenario = load(args.scenario_name + ".py").NvSymmetry(args.n_rollout_threads)
    # this is a vector environment
    env = MultiAgentUAVEnv(scenario)
    return env 


def parse_args(parser):
    parser.add_argument("--scenario_name", type=str,
                        default="N_v_Default1", 
                        help="which scenario to run on.")
    
    parser.add_argument("--n_rollout_threads", type=int,
                        default=10, 
                        help="number of rollout threads")
                        
    all_args = parser.parse_known_args()

    return all_args


def main(args=None):
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)
    all_args = parse_args(parser)
    env = UAVEnv(all_args)
    env.reset()
    for i in range(1000):
        print("Testing-----")
        start = time.time()
        act = np.zeros((64, 8), dtype=np.float32)
        obs, rew, don, inf  =env.step(act)
        end = time.time() 
        print("Testing stepping takes ", end - start)
        print("obs={}".format(obs))


if __name__ == "__main__":
    main()