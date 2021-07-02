import agent_registry, env_registry
import ExperimentModel
import pickle
import joblib
import tensorflow as tf
from rllab.misc.instrument import run_experiment_lite
import Box2D
import json
import glob
import numpy as np
import logging
import socket
import time
import shutil
import sys
from copy import deepcopy
from pprint import pprint
import os
sys.path.append(os.getcwd())


def exp_name(perm, run):
    hps = deepcopy(perm["metaParameters"])
    hps["n_itr"] = 500
    agent = perm["agent"]
    env = perm["problem"]
    hp_list = list(map(lambda x: str(x), hps.values()))
    hp_list = "-".join(hp_list)
    return agent + "," + env + "," + hp_list + f"Run{run}"


def first_int(filename):
    for i in range(len(filename)):
        if filename[i].isdigit():
            return i
    return None


def get_algo(perm, env, run):
    name = exp_name(perm, run)
    dir_ = f"./data/local/experiment/{name}"
    if os.path.exists(os.path.join(dir_, "params.pkl")):
        algo = joblib.load(os.path.join(dir_, "params.pkl"))["algo"]
        algo.n_itr = perm["metaParameters"]["n_itr"]

        if perm["agent"] == "reinforce":
            with open(os.path.join(dir_, "progress0.csv"), "r") as infile:
                completed_itr = int(infile.readlines()[-1].split(",")[0])
                print("COMPLETED:", completed_itr)
            algo.current_itr = completed_itr + 1

        print("Current itr:", algo.current_itr, "\tN itr:", algo.n_itr)
    else:
        algo = agent_registry.get(perm, env)

    return algo


def run_task(*_):
    env = env_registry.get(perm)

    algo = get_algo(perm, env, run)

    algo.train()


def progress_file(filename):
    return "progress" in filename


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('run again with:')
        print('python3 src/main.py <path/to/description.json> <idx> <opt:prod>')
        exit(1)

    prod = len(sys.argv) == 4 or 'cdr' in socket.gethostname()
    # prod = True
    if not prod:
        logging.basicConfig(level=logging.DEBUG)
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)

    exp = ExperimentModel.load(sys.argv[1])
    idx = int(sys.argv[2])
    base = './' if len(sys.argv) >= 3 else sys.argv[3]

    # Get hyperparameters for this run
    perm = exp.getPermutation(idx)
    hps = perm["metaParameters"]
    pprint(perm)

    # Get the maximum number of steps for training and the run number
    max_steps = exp.max_steps
    run = exp.getRun(idx)
    print("Run:", run)

    # set random seeds accordingly
    np.random.seed(run)
    tf.random.set_seed(run)

    # Copy old progress file so that we can continue running
    name = exp_name(perm, run)
    dir_ = f"./data/local/experiment/{name}"
    if os.path.exists(dir_):
        files = list(filter(progress_file, os.listdir(dir_)))
        print(files)

        for file_ in files:

            ind = first_int(file_)
            print("INDEX:", ind)

            if ind is not None:
                suffix = str(int(file_[ind:len(file_)-4]) + 1)
                print("SUFFIX:", suffix)
            else:
                suffix = "0"

            with open(os.path.join(dir_, file_), "r") as old:
                data = old.readlines()
            with open(os.path.join(dir_, "progress" + suffix + ".csv"), "w") as new:
                new.writelines(data)

    start = time.time()
    run_experiment_lite(
        run_task,
        n_parallel=1,
        snapshot_mode="last",
        seed=run,
        exp_name=exp_name(perm, run),
    )
    cpu_time = time.time() - start

    if not os.path.exists(f"./data/local/experiment/{exp_name(perm, run)}"):
        os.mkdir(f"./data/local/experiment/{exp_name(perm)}")

    file_ = f"./data/local/experiment/{exp_name(perm, run)}/config.json"
    with open(file_, "w") as outfile:
        json.dump(perm, outfile, indent=4)

    file_ = f"./data/local/experiment/{exp_name(perm, run)}/cpu_time"
    with open(file_, "w") as outfile:
        outfile.write(str(cpu_time))

    # Combine progress files
    name = exp_name(perm, run)
    dir_ = f"./data/local/experiment/{name}"
    if os.path.exists(dir_):
        files = sorted(list(filter(progress_file, os.listdir(dir_))))[::-1]
        print("Combining:", files)
        data = []
        for file_ in files:
            with open(os.path.join(dir_, file_), "r") as infile:
                if file_ == files[0]:
                    data.extend(infile.readlines())
                else:
                    data.extend(infile.readlines()[1:])
            os.remove(os.path.join(dir_, file_))

        with open(os.path.join(dir_, "progress.csv"), "w") as outfile:
            outfile.writelines(data)

