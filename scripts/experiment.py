#!/usr/bin/env python
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

import time

import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

DATA_DIR_KEYS = {"cost_limit": "cost"}
DATA_DIR_SUFFIX = "_benchmark_mf"

DATA_DIR = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), "data")


def gen_data_dir_name(env, cost):
    name = env + '_' + "cost" + '_' + str(cost)
    return name + DATA_DIR_SUFFIX

def main(robot, task, algo, seed, cpu, cost):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo', 'ball']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'circle', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    exp_name = algo + '_' + robot + task
    if task == 'Circle':
        env_name = 'Safety'+robot+task+'-v0'
        max_ep_len = 300
    else:
        env_name = 'Safexp-'+robot+task+'-v0'
        max_ep_len = 400

    # Hyperparameters
    num_steps = 2e6
    steps_per_epoch = 4000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = cost

    # Fork for parallelizing
    mpi_fork(cpu)

    exp_name = algo

    # Prepare Logger
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
         target_kl=target_kl,
         max_ep_len=max_ep_len,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs
         )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--cost', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    main(args.robot, args.task, args.algo, args.seed, args.cpu, args.cost)