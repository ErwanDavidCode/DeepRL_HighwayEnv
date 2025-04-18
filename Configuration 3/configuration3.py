# Erwan DAVID - Guillaume FAYNOT

import pickle
import gymnasium as gym
import highway_env
from highway_env.envs import IntersectionEnv

# gym.register_envs(highway_env)


config_dict = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "absolute": True,
        "flatten": False,
        "observe_intentions": False
    },
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": False,
        "lateral": True
    },
    "duration": 13,  # [s]
    "destination": "o1",
    "initial_vehicle_count": 10,
    "spawn_probability": 0.6,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "collision_reward": IntersectionEnv.default_config()['collision_reward'],
    "normalize_reward": False
}

with open("config.pkl", "wb") as f:
    pickle.dump(config_dict, f)

# env = gym.make("intersection-v0", render_mode="rgb_array")
# env.unwrapped.configure(config_dict)
# print(env.reset())
