from rllab.envs.normalized_env import normalize
import Box2D


def get(perm):
    name = perm["problem"]
    if name.lower() == "cartpole":
        from rllab.envs.box2d.cartpole_env import CartpoleEnv
        return normalize(CartpoleEnv())

    elif name.lower() == "mountain car height bonus":
        from rllab.envs.box2d.mountain_car_env import MountainCarEnv
        return normalize(MountainCarEnv())

    elif name.lower() == "mountain car":
        from rllab.envs.box2d.mountain_car_env import MountainCarEnv
        return normalize(MountainCarEnv(height_bonus=0))

    elif name.lower() == "gym mountain car":
        from rllab.envs.gym_env import GymEnv
        return normalize(GymEnv("MountainCarContinuous-v0",
                                record_video=False))

    elif name.lower() == "pendulum":
        from rllab.envs.gym_env import GymEnv
        return normalize(GymEnv("Pendulum-v0", record_video=False))

    elif name.lower() == "double pendulum":
        from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
        return normalize(DoublePendulumEnv())

    elif name.lower() == "hopper":
        from rllab.envs.mujoco.hopper_env import HopperEnv
        return normalize(HopperEnv())

    elif name.lower() == "swimmer":
        from rllab.envs.mujoco.swimmer_env import SwimmerEnv
        return normalize(SwimmerEnv())

    elif name.lower() == "2d walker":
        from rllab.envs.mujoco.walker2d_env import Walker2DEnv
        return normalize(Walker2DEnv())

    elif name.lower() == "half cheetah":
        from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
        return normalize(HalfCheetahEnv())

    elif name.lower() == "ant":
        from rllab.envs.mujoco.ant_env import AntEnv
        return normalize(AntEnv())

    elif name.lower() == "simple humanoid":
        from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
        return normalize(SimpleHumanoidEnv())

    elif name.lower() == "full humanoid":
        from rllab.envs.mujoco.humanoid_env import HumanoidEnv
        return normalize(HumanoidEnv())

    else:
        raise NotImplementedError(f"Environment {name} unknown")
