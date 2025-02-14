from rllab.algos.ddpg import DDPG
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.algos.vpg import VPG
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import sys
import os
sys.path.append(os.getcwd())


def get(perm, env):
    name = perm["agent"]
    hps = perm["metaParameters"]
    if name.lower() == "ddpg":
        return _create_ddpg(hps, env)

    elif name.lower() == "trpo":
        return _create_trpo(hps, env)

    elif name.lower() == "reinforce":
        return _create_reinforce(hps, env)

    else:
        raise NotImplementedError("Agent", perm["agent"], "unknown")


def _create_reinforce(hps, env):
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    policy_weights = hps["policy_weights"]
    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=policy_weights)
    stepsize = hps["stepsize"]

    n_itr = hps["n_itr"]
    max_path_length = hps["max_path_length"]
    discount = hps["discount"]
    batch_size = hps["batch_size"]

    optimizer_args = {"learning_rate": stepsize,
                      "batch_size": None,  # The whole episode is used
                      }

    return VPG(env, policy, baseline=baseline, discount=discount, n_itr=n_itr,
               batch_size=batch_size, max_path_length=max_path_length,
               optimizer_args=optimizer_args)


def _create_trpo(hps, env):
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    policy_weights = hps["policy_weights"]
    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=policy_weights)
    stepsize = hps["stepsize"]

    batch_size = hps["batch_size"]
    n_itr = hps["n_itr"]
    max_path_length = hps["max_path_length"]
    discount = hps["discount"]

    return TRPO(env=env, policy=policy, baseline=baseline,
                max_path_length=max_path_length, discount=discount,
                step_size=stepsize, n_itr=n_itr, batch_size=batch_size)


def _create_ddpg(hps, env):
    batch_size = hps["batch_size"]
    replay_cap = hps["replay_pool_size"]
    epoch_length = hps["epoch_length"]
    epochs = hps["n_epochs"]
    discount = hps["discount"]

    critic_lr = hps["qf_learning_rate"]
    actor_lr = critic_lr * hps["policy_learning_scale"]
    max_path_length = hps["max_path_length"]

    soft_target = hps["target_update"][1] == 1
    target_update_steps = hps["target_update"][1]
    tau = hps["target_update"][0]

    n_updates_per_sample = hps["n_updates_per_sample"]

    policy_weights = hps["actor_weights"]
    policy = DeterministicMLPPolicy(env_spec=env.spec,
                                    hidden_sizes=policy_weights)

    critic_weights = hps["critic_weights"]
    qf = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=critic_weights)

    sigma = hps["sigma"]
    es = OUStrategy(env_spec=env.spec, sigma=sigma)

    return DDPG(env, policy, qf, es, batch_size, epochs, epoch_length,
                replay_pool_size=replay_cap, discount=discount,
                max_path_length=max_path_length, qf_learning_rate=critic_lr,
                policy_learning_rate=actor_lr, soft_target=soft_target,
                soft_target_tau=tau, target_update_steps=target_update_steps,
                n_updates_per_sample=n_updates_per_sample,
                min_pool_size=batch_size+1)
