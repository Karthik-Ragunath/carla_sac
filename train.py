import argparse
import numpy as np
from parl.utils import logger, tensorboard
from replay_memory import ReplayMemory
from env_utils import Env, LocalEnv
from torch_base import TorchModel, TorchSAC, TorchAgent  # Choose base wrt which deep-learning framework you are using
import torch
import os
from env_config import EnvConfig
import matplotlib.pyplot as plt
from torch_base import DetectBoundingBox
import glob
import shutil

WARMUP_STEPS = 60
EVAL_EPISODES = 1
# MEMORY_SIZE = int(1e4)
# BATCH_SIZE = 256
BATCH_SIZE = 30
MEMORY_SIZE = 60
SEQUENTIAL_SIZE = 30
GAMMA = 0.99
TAU = 0.005
# ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ALPHA = 0.01
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array

def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

# Runs policy for 3 episodes by default and returns average reward
def run_evaluate_episodes(agent: TorchAgent, env: Env, eval_episodes, valid_vis_dir):
    avg_reward = 0.
    env.eval_episode_count += 1
    env.env.eval_episode_num = env.eval_episode_count
    dir_path = os.path.join(os.getcwd(), valid_vis_dir, str(env.eval_episode_count))
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    for k in range(eval_episodes):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < env._max_episode_steps:
            steps += 1
            action = agent.predict(obs)
            if env.eval_episode_count % 5 == 0:
                reward, done, next_obs_rgb = env.step(action, is_validation=True)
            else:
                reward, done, next_obs_rgb = env.step(action, is_validation=False)
            avg_reward += reward
            obs = next_obs_rgb
    avg_reward /= eval_episodes
    return avg_reward, steps


def main():
    logger.info("-----------------Carla_SAC-------------------")
    logger.set_dir('./{args_env}_train_{train_context}'.format(args_env=args.env, train_context=EnvConfig['train_context']))

    # Parallel environments for training
    train_envs_params = EnvConfig['train_envs_params']
    train_env = Env(args.env, train_envs_params, EnvConfig['train_context'])

    # env for eval
    eval_env_params = EnvConfig['eval_env_params']
    eval_env = Env(args.env, eval_env_params, EnvConfig['train_context'])
    obs_dim = eval_env.obs_dim
    action_dim = eval_env.action_dim

    train_vis_dir = EnvConfig['train_context'] + '_train'
    valid_vis_dir = EnvConfig['train_context'] + '_valid'

    if not os.path.exists(train_vis_dir):
        os.makedirs(train_vis_dir)
    
    if not os.path.exists(valid_vis_dir):
        os.makedirs(valid_vis_dir)

    # Initialize model, algorithm, agent, replay_memory
    CarlaModel, SAC, CarlaAgent = TorchModel, TorchSAC, TorchAgent
    model = CarlaModel(obs_dim, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    pretrained_steps = 0
    if train_envs_params.get('load_recent_model', False):
        # set the computation device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        filenames = glob.glob("torch_model_rgb_actor_critic_modified/*.ckpt")
        model_filename = None
        max_train_epoch = 0
        for filename in filenames:
            epoch_num = int(filename.split('/')[-1].split('_')[1])
            if epoch_num > max_train_epoch:
                max_train_epoch = epoch_num
                model_filename = filename
                pretrained_steps = max_train_epoch
        if model_filename:
            model.load_state_dict(torch.load(
                os.path.join(os.getcwd(), model_filename), map_location=device
            ))
    agent = CarlaAgent(algorithm)
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

    total_steps = 0
    last_save_steps = 0
    test_flag = 0
    obs = train_env.reset()
    while total_steps < args.train_total_steps:
        # Train episode
        if rpm.size() < WARMUP_STEPS:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            action = agent.sample(obs)

        reward, done, next_obs_rgb = train_env.step(action)

        rpm.append(obs, action, reward, next_obs_rgb, done)

        obs = train_env.get_obs()
        total_steps = train_env.total_steps

        #logger.info('----------- Step 1 ------------')
        # Train agent after collecting sufficient data
        if rpm.size() >= WARMUP_STEPS:
            # batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
            #     BATCH_SIZE)
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_sequentially(BATCH_SIZE, sequential_size=SEQUENTIAL_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)
        # print("-------------------------")

        # logger.info('----------- Step 2 ------------')
        # Save agent
        # if total_steps > int(1e5) and total_steps > last_save_steps + int(1e4):
        if total_steps > int(WARMUP_STEPS) and total_steps > last_save_steps + int(500):
            agent.save('./{model_framework}_model_{train_context}/step_{current_steps}_model.ckpt'.format(
                model_framework=args.framework, current_steps=(total_steps + pretrained_steps), train_context=EnvConfig['train_context']))
            last_save_steps = total_steps
        
        
        #logger.info('----------- Step 3 ------------')
        # Evaluate episode
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            # while (total_steps + 1) // args.test_every_steps >= test_flag:
            #     test_flag += 1
            test_flag += 1
            avg_reward, num_steps = run_evaluate_episodes(agent, eval_env, EVAL_EPISODES, valid_vis_dir)
            tensorboard.add_scalar('eval/episode_reward', avg_reward,
                                   total_steps + pretrained_steps)
            tensorboard.add_scalar('eval/episode_steps', num_steps,
                                    total_steps + pretrained_steps)
            logger.info(
                'Total steps {}, Evaluation over {} episodes, Average reward: {}, Episode steps: {}'
                .format(total_steps + pretrained_steps, EVAL_EPISODES, avg_reward, num_steps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xparl_addr",
        default='localhost:8080',
        help='xparl address for parallel training')
    parser.add_argument("--env", default="carla-v0")
    parser.add_argument(
        '--framework',
        default='paddle',
        help='choose deep learning framework: torch or paddle')
    parser.add_argument(
        "--train_total_steps",
        default=5e5,
        type=int,
        help='max time steps to run environment')
    parser.add_argument(
        "--test_every_steps",
        default=1e3,
        type=int,
        help='the step interval between two consecutive evaluations')
    args = parser.parse_args()

    main()
