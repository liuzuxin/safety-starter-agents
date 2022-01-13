import subprocess



def experiment_circle():
    robot_list = ['car', 'ant']
    task_list = ['circle']
    algo_list = ['ppo_lagrangian', 'trpo_lagrangian', 'cpo', 'trpo']

    # for algo in algo_list:
    task = task_list[0]

    p_list = []
    for robot in robot_list:
        for cost in [20]:
            for algo in algo_list:
                for seed in [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]:
                    print("Robot: ", robot, " Algo: ", algo, " Seed: ", seed, " Cost: ", cost)
                    command = ['python'] + ['scripts/experiment.py'] + ['--robot'] + [robot] + ['--task'] + [task] + \
                        ['--algo'] + [algo] + ['--cost'] + [str(cost)] + ['--seed'] + [str(seed)]
                    p = subprocess.Popen(command)
                    p_list.append(p)

                global pid_list
                pid_list = [p.pid for p in p_list]

                for p in p_list:
                    p.wait()

                import os
                import signal
                def kill_child():
                    global pid_list
                    for pid in pid_list:
                        if pid is None:
                            pass
                        else:
                            try:
                                os.kill(pid, signal.SIGTERM)
                            except OSError:
                                pass
                            else:
                                pass

                import atexit
                atexit.register(kill_child)


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--algo','-a', type=str, default='cpo')
    # args = parser.parse_args()
    experiment_circle()