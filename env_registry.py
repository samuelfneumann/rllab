from rllab.envs.normalized_env import normalize
import Box2D

def get(name):
    if name.lower() == "cartpole":
        from rllab.envs.box2d.cartpole_env import CartpoleEnv
        return normalize(CartpoleEnv())

    elif name.lower() == "mountain car":
        from rllab.envs.box2d.mountain_car_env import MountainCarEnv
        return normalize(MountainCarEnv())

    elif name.lower() == "gym mountain car":
        from rllab.envs.gym_env import GymEnv
        return normalize(GymEnv("MountainCarContinuous-v0"))

    elif name.lower() == "pendulum":
        from rllab.envs.gym_env import GymEnv
        return normalize(GymEnv("Pendulum-v0"))

    elif name.lower() == "acrobot":
        from rllab.envs.gym_env import GymEnv
        return normalize(GymEnv("Acrobot-v1"))

    else:
        raise NotImplementedError(f"Environment {name} unknown")
