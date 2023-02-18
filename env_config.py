from copy import deepcopy

# set max_episode_steps according to task_mode
# e.g. task_model   max_episode_steps
#      Lane         250
#      Long         200
TASK_MODE = 'Lane'
MAX_EPISODE_STEPS = 3000
OPEN_AI_IMPLEMENTATION = False

params = {
    'action_dim': 3 if OPEN_AI_IMPLEMENTATION else 2,
    # screen size of cv2 window
    'obs_size': (160, 100),
    # time interval between two frames
    'dt': 0.025,
    # filter for defining ego vehicle
    'ego_vehicle_filter': 'vehicle.lincoln*',
    # filter for defining traffic vehicles
    'traffic_vehicle_filter': 'vehicle.bmw.*',
    # CARLA service's port
    'port': 2000,
    # mode of the task, [random, roundabout (only for Town03)]
    'task_mode': TASK_MODE,
    # mode of env (test/train)
    'code_mode': 'test',
    # maximum timesteps per episode
    'max_time_episode': MAX_EPISODE_STEPS,
    # desired speed (m/s)
    'desired_speed': 15,
    # maximum times to spawn ego vehicle
    'max_ego_spawn_times': 100,
    # maximum number of traffic vehicles
    'max_traffic_vehicles': 1,
    # sensor tick,
    'sensor_tick': 0.5
}

# train env params
train_env_port = 2021
train_code_mode = 'train'
train_envs_params = deepcopy(params)
train_envs_params['port'] = train_env_port
train_envs_params['code_mode'] = train_code_mode
train_envs_params['load_recent_model'] = False
train_envs_params['image_dimensions'] = (96, 96)

# evaluate env params
eval_port = 2027
eval_code_mode = 'test'
temp_params = deepcopy(params)
temp_params['port'] = eval_port
temp_params['code_mode'] = eval_code_mode
eval_env_params = temp_params
eval_env_params['load_recent_model'] = True
eval_env_params['image_dimensions'] = (96, 96)

# test env params
test_port = 2029
test_code_mode = 'test'
temp_params = deepcopy(params)
temp_params['port'] = test_port
temp_params['code_mode'] = test_code_mode
test_env_params = temp_params
test_env_params['load_recent_model'] = True
test_env_params['image_dimensions'] = (96, 96)

EnvConfig = {
    # train envs config
    'train_envs_params': train_envs_params,
    'train_env_params': train_envs_params,

    # eval env config
    'eval_env_params': eval_env_params,

    # env config for evaluate.py
    'test_env_params': test_env_params
}
