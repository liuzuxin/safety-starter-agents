import os
import os.path as osp
os.environ["KMP_WARNINGS"] = "off" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tf log errors only

import ray
from ray import tune

DATA_DIR = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), "data")

ENV_LIST = [
    'SafetyCarCircle-v0',
    'SafetyBallCircle-v0',
]

EXP_CONFIG = dict(
    env=tune.grid_search(ENV_LIST),
    policy=tune.grid_search(["ppo_lagrangian", 'trpo_lagrangian', 'cpo']),
    cost_limit=tune.grid_search([4, 20]),
    seed=tune.grid_search([0, 11, 22]),
)

EXP_NAME_KEYS = {}
DATA_DIR_KEYS = {"cost_limit": "cost"}
DATA_DIR_SUFFIX = "_benchmark_mf"


def gen_exp_name(config: dict):
    name = config["policy"]
    for k in EXP_NAME_KEYS:
        name += '_' + EXP_NAME_KEYS[k] + '_' + str(config[k])
    return name


def gen_data_dir_name(config: dict):
    name = config["env"]
    for k in DATA_DIR_KEYS:
        name += '_' + DATA_DIR_KEYS[k] + '_' + str(config[k])
    return name + DATA_DIR_SUFFIX

def trial_name_creator(trial):
    config = trial.config
    name = config["env"]
    for k in DATA_DIR_KEYS:
        name += '_' + DATA_DIR_KEYS[k] + '_' + str(config[k])
    return name + DATA_DIR_SUFFIX + '_' + config["policy"]

def trainable(config):
    import os
    import os.path as osp
    os.environ["KMP_WARNINGS"] = "off" 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tf log errors only
    from safe_rl import ppo, ppo_lagrangian, trpo, trpo_lagrangian, cpo
    import gym 
    import safety_gym
    import bullet_safety_gym
    import safe_rl
    from safe_rl.utils.run_utils import setup_logger_kwargs
    # from safe_rl.utils.mpi_tools import mpi_fork
    seed = config["seed"]
    policy = config["policy"]
    env_name = config["env"]
    cost_lim = config["cost_limit"]

    num_steps = 5e6
    steps_per_epoch = 5000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    

    # Fork for parallelizing
    # mpi_fork(0)

    # Prepare Logger
    exp_name = gen_exp_name(config)

    data_dir = osp.join(DATA_DIR, gen_data_dir_name(config))
    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=data_dir)

    # Algo and Env
    algo = eval(policy)

    if env_name == 'SafetyCarCircle-v0' or env_name == 'SafetyBallCircle-v0':
        max_ep_len = 300
    else:
        max_ep_len = 400

    algo(env_fn=lambda: gym.make(env_name),
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         max_ep_len=max_ep_len,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs
         )          


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', '-e', type=str, default='exp', help='runner or kl')
    parser.add_argument('--cpus',
                        '--cpu',
                        type=int,
                        default=16,
                        help='maximum cpu resources for ray')
    parser.add_argument('--threads',
                        '--thread',
                        type=int,
                        default=1,
                        help='maximum threads resources per trial')

    args = parser.parse_args()

    ray.init(num_cpus=args.cpus)

    EXP_CONFIG["threads"] = args.threads

    experiment_spec = tune.Experiment(
        args.exp,
        trainable,
        config=EXP_CONFIG,
        resources_per_trial={
            "cpu": args.threads,
            "gpu": 0
        },
        trial_name_creator=trial_name_creator,
    )

    tune.run_experiments(experiment_spec)