import os
import os.path as osp
os.environ["KMP_WARNINGS"] = "off" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tf log errors only
import gym 
import safety_gym
import bullet_safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork

EXP_NAME_KEYS = {}
DATA_DIR_KEYS = {"cost_limit": "cost"}
DATA_DIR_SUFFIX = "_benchmark_mf"

DATA_DIR = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), "data")


def gen_data_dir_name(env, cost):
    name = env + '_' + "cost" + '_' + str(cost)
    return name + DATA_DIR_SUFFIX

def train(robot, task, cpu, seed, algo, cost_lim = 20):
    # Hyperparameters
    exp_name = algo + '_' + robot + task
    if task == 'Circle':
        env_name = 'Safety'+robot+task+'-v0'
        max_ep_len = 300
    else:
        env_name = 'Safexp-'+robot+task+'-v0'
        max_ep_len = 400

    num_steps = 2e6
    steps_per_epoch = 5000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    exp_name = algo
    data_dir = osp.join(DATA_DIR, gen_data_dir_name(env_name, cost_lim))
    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=data_dir)

    # Algo and Env
    algo = eval('safe_rl.'+algo)

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

def main(robot, task, cpu):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo', 'ball']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2', 'circle']
    algo_list = ['ppo_lagrangian', 'trpo_lagrangian', 'cpo']

    task = task.capitalize()
    robot = robot.capitalize()
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    for algo in algo_list:
        for cost in [10, 30]:
            for seed in [0, 11, 22]:
                print("Algo: ", algo)
                print("seed: ", seed)
                train(robot, task, cpu, seed, algo, cost)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', '-r', type=str, default='car')
    parser.add_argument('--task', '-t', type=str, default='button1')
    parser.add_argument('--cpu', type=int, default=2)
    args = parser.parse_args()
    main(args.robot, args.task, args.cpu)