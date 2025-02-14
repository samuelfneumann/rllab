import time
from copy import deepcopy
import sys
import os
sys.path.append(os.getcwd() + '/src')
import math
import numpy as np
import PyExpUtils.runner.Slurm as Slurm
import PyExpUtils.runner.parallel as Parallel
from PyExpUtils.results.backends.h5 import detectMissingIndices
from PyExpUtils.utils.generator import group
import ExperimentModel

if len(sys.argv) < 4:
    print('Please run again using')
    print('python scripts/scriptName.py [path/to/slurm-def] [src/executable.py] [base_path] [runs] [paths/to/descriptions]...')
    exit(0)

# -------------------------------
# Generate scheduling bash script
# -------------------------------

# the contents of the string below will be the bash script that is scheduled on compute canada
# change the script accordingly (e.g. add the necessary `module load X` commands)
cwd = os.getcwd()
def getJobScript(parallel):
    return f"""#!/bin/bash
cd {cwd}
module load python/3.7
module load cuda
module load cudnn/8.0.3

. ~/venv3_7/bin/activate

nvidia-smi
export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1
{parallel}
    """

# --------------------------
# Get command-line arguments
# --------------------------
slurm_path = sys.argv[1]
executable = sys.argv[2]
base_path = sys.argv[3]
runs = int(sys.argv[4])
experiment_paths = sys.argv[5:]

# prints a progress bar
def printProgress(size, it):
    for i, _ in enumerate(it):
        print(f'{i + 1}/{size}', end='\r')
        if i - 1 == size:
            print()
        yield _

def exp_name(perm, run):
    hps = deepcopy(perm["metaParameters"])
    hps["n_itr"] = 500
    agent = perm["agent"]
    env = perm["problem"]
    hp_list = list(map(lambda x: str(x), hps.values()))
    hp_list = "-".join(hp_list)
    return agent + "," + env + "," + hp_list + f"Run{run}"


def estimateUsage(indices, groupSize, cores, hours):
    jobs = math.ceil(len(indices) / groupSize)

    total_cores = jobs * cores
    core_hours = total_cores * hours

    core_years = core_hours / (24 * 365)
    allocation = 724

    return core_years, 100 * core_years / allocation

def gatherMissing(experiment_paths, runs, groupSize, cores, total_hours):
    out = {}

    approximate_cost = np.zeros(2)
    indices = []

    for path in experiment_paths:
        exp = ExperimentModel.load(path)

        # Find the number of permutations
        firstPerm = exp.getPermutation(0)
        numPerms = 0
        perm = None
        while perm != firstPerm:
            numPerms += 1
            perm = exp.getPermutation(numPerms)

        for i in range(numPerms * runs):
            perm = exp.getPermutation(i)
            run = i // numPerms
            name = exp_name(perm, run)
            dataPath = f"./data/local/experiment/{name}"

            if not os.path.exists(dataPath):
                print(dataPath)
                indices.append(i)
                continue

            with open(os.path.join(dataPath, "progress.csv")) as infile:
                #if not os.path.exists(f"./data/local/experiment/{name}/cpu_time"):
                lines = infile.readlines()
                if len(lines) < 500001:
                    print(path, i, dataPath)
                    indices.append(i)

        indices = sorted(indices)
        out[path] = indices

        approximate_cost += estimateUsage(indices, groupSize, cores,
                                          total_hours)

        # figure out how many indices to expect
        size = exp.numPermutations() * runs

        # log how many are missing
        print(path, f'{len(indices)} / {size}')

    return out, approximate_cost

# ----------------
# Scheduling logic
# ----------------
slurm = Slurm.fromFile(slurm_path)

# compute how many "tasks" to clump into each job
groupSize = slurm.cores * slurm.sequential

# compute how much time the jobs are going to take
hours, minutes, seconds = slurm.time.split(':')
total_hours = int(hours) + (int(minutes) / 60) + (int(seconds) / 3600)

# gather missing and sum up cost
missing, cost = gatherMissing(experiment_paths, runs, groupSize, slurm.cores, total_hours)

print(f"Expected to use {cost[0]:.2f} core years, which is {cost[1]:.4f}% of our annual allocation")
input("Press Enter to confirm or ctrl+c to exit")

for path in missing:
    # reload this because we do bad mutable things later on
    slurm = Slurm.fromFile(slurm_path)

    for g in group(missing[path], groupSize):
        l = list(g)

        # build the executable string
        runner = f'python {executable} {path} '
        # generate the gnu-parallel command for dispatching to many CPUs across server nodes
        parallel = Parallel.build({
            'executable': runner,
            'cores': slurm.cores,
            'tasks': l,
        })

        # generate the bash script which will be scheduled
        # we need it in the form parallel 'CUDA_VISIBLE_DEVICES=$(({%} - 1)); python src/dm_control.py experiments/benchmark/DMCartPole/DDPG/json {}' ::: 0 1 2 3 4 5 ...
        i = parallel.index("python")
        j = parallel.index(":")
        #parallel = parallel[:i] + " 'CUDA_VISIBLE_DEVICES=$(({%} - 1)); " + parallel[i:j-1] + "{}' " + parallel[j:]
        print(parallel)
        script = getJobScript(parallel)

        ## uncomment for debugging the scheduler to see what bash script would have been scheduled
        print(script)
        #exit()

        # make sure to only request the number of CPU cores necessary
        slurm.cores = min([slurm.cores, len(l)])
        Slurm.schedule(script, slurm)

        # DO NOT REMOVE. This will prevent you from overburdening the slurm scheduler. Be a good citizen.
        time.sleep(2)
