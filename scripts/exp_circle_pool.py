import subprocess
from multiprocessing import Pool

from experiment import *

def exp_one_config(robot, algo, seed, cost, task):
    print("Robot: ", robot, " Algo: ", algo, " Seed: ", seed, " Cost: ", cost)
    command = ['python'] + ['scripts/experiment.py'] + ['--robot'] + [robot] + ['--task'] + [task] + \
        ['--algo'] + [algo] + ['--cost'] + [str(cost)] + ['--seed'] + [str(seed)]
    p = subprocess.Popen(command)
    p.wait()
    return p.pid


def experiment_button(algo):
    robot_list = ['car', 'ball']
    task_list = ['circle']
    algo_list = ['ppo_lagrangian', 'trpo_lagrangian', 'cpo']
    # for algo in algo_list:
    algo = algo
    task = task_list[0]

    for robot in robot_list:
        args_list = []
        for cost in [4, 20]:
            for seed in [0, 11, 22]:
                args = (robot, algo, seed, cost, task)
                args_list.append(args)
        pool = Pool(6)
        global pid_list
        pid_list = pool.starmap(exp_one_config, args_list)

        import os
        import signal
        def kill_child():
            global pid_list
            for pid in pid_list:
                if pid is None:
                    pass
                else:
                    os.kill(pid, signal.SIGTERM)

        import atexit
        atexit.register(kill_child)

        pool.close()
        pool.join()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo','-a', type=str, default='cpo')
    args = parser.parse_args()
    experiment_button(args.algo)