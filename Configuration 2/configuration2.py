# Erwan DAVID - Guillaume FAYNOT

import pickle
# import gymnasium as gym
# import highway_env

# gym.register_envs(highway_env)

config_dict = {
    "observation": {
        "type": "OccupancyGrid",
        "features": ['presence', 'on_road'],
        "grid_size": [[-21, 21], [-10.3, 10.7]],
        "grid_step": [6, 3],

        "as_image": False,
        "align_to_vehicle_axes": True
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": False,
        "lateral": True
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 300,
    "collision_reward": -30,
    #"lane_centering_cost": 0.2,
    "action_reward": -0.5, #-0.3,
    "on_road_reward": 0.5,
    "no_collision_reward": 10,
    #off_road_reward": -3,
    "controlled_vehicles": 1,
    "other_vehicles": 10,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}

with open("config.pkl", "wb") as f:
    pickle.dump(config_dict, f)

# env = gym.make("racetrack-v0", render_mode="rgb_array")
# env.unwrapped.configure(config_dict)
# print(env.reset())
