from gym.envs.registration import register
register(
    id='carla_environment-v0',
    entry_point='gym_carla_environment.envs:CarlaEnv',
)
