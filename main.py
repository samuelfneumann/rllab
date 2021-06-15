import agent_registry, env_registry
import ExperimentModel
import tensorflow as tf
from rllab.misc.instrument import run_experiment_lite
import Box2D
import json
import numpy as np
import logging
import socket
import time
import sys
from pprint import pprint
import os
sys.path.append(os.getcwd())

# TODO:
#   target net averaging


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
if perm["max_steps"] != hps["n_epochs"] * hps["epoch_length"]:
    raise ValueError("max steps is not set to equal epochs * epoch length")

# Get the maximum number of steps for training and the run number
max_steps = exp.max_steps
run = exp.getRun(idx)
print("Run:", run)

# set random seeds accordingly
np.random.seed(run)
tf.random.set_seed(run)


def exp_name(perm):
    hps = perm["metaParameters"]
    agent = perm["agent"]
    env = perm["problem"]
    hp_list = list(map(lambda x: str(x), hps.values()))
    hp_list = ",".join(hp_list)
    return agent + "," + env + "," + hp_list


def run_task(*_):
    env = env_registry.get(perm["problem"])

    algo = agent_registry.get(perm, env)

    algo.train()


start = time.time()
run_experiment_lite(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=run,
    exp_name=exp_name(perm)
)
cpu_time = time.time() - start

if not os.path.exists(f"./data/local/experiment/{exp_name(perm)}"):
    os.mkdir(f"./data/local/experiment/{exp_name(perm)}")

with open(f"./data/local/experiment/{exp_name(perm)}/config.json", "w") as outfile:
    json.dump(perm, outfile, indent=4)

with open(f"./data/local/experiment/{exp_name(perm)}/cpu_time", "w") as outfile:
    outfile.write(str(cpu_time))
