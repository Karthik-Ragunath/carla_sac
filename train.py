import argparse
import numpy as np
from parl.utils import logger, tensorboard, ReplayMemory
from env_utils import ParallelEnv, LocalEnv
from torch_base import TorchModel, TorchSAC, TorchAgent  # Choose base wrt which deep-learning framework you are using
#from paddle_base import PaddleModel, PaddleSAC, PaddleAgent
from env_config import EnvConfig
import matplotlib.pyplot as plt
from torch_base import DetectBoundingBox

WARMUP_STEPS = 2e3
EVAL_EPISODES = 3
# MEMORY_SIZE = int(1e4)
# BATCH_SIZE = 256
BATCH_SIZE = 5
MEMORY_SIZE = 10
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
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
def run_evaluate_episodes(agent, env, eval_episodes):
    avg_reward = 0.
    for k in range(eval_episodes):
        obs, train_image = env.reset()
        if train_image:
            print("*" * 25, "RUN EPISODE - RESET", "*" * 25)
            numpy_rgb_image = to_rgb_array(train_image)
            plt.imshow(numpy_rgb_image)
            plt.savefig("carla_rgb_sensor_flow_detected/" + str(train_image.frame) + '.png')
            detect_bounding_box_obj = DetectBoundingBox(numpy_rgb_image.copy(), str(train_image.frame))
            bounding_box_image = detect_bounding_box_obj.detect_bounding_boxes()
        done = False
        steps = 0
        while not done and steps < env._max_episode_steps:
            steps += 1
            action = agent.predict(numpy_rgb_image, bounding_box_image)
            step_tuple, train_image = env.step(action)
            if train_image:
                numpy_rgb_image = to_rgb_array(train_image)
                plt.imshow(numpy_rgb_image)
                plt.savefig("carla_rgb_sensor_flow_detected/" + str(train_image.frame) + '.png')
                detect_bounding_box_obj = DetectBoundingBox(numpy_rgb_image.copy(), str(train_image.frame))
                bounding_box_image = detect_bounding_box_obj.detect_bounding_boxes()
            obs, reward, done, _ = step_tuple
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def main():
    logger.info("-----------------Carla_SAC-------------------")
    logger.set_dir('./{args_env}_train_{train_context}'.format(args_env=args.env, train_context=EnvConfig['train_context']))

    # Parallel environments for training
    train_envs_params = EnvConfig['train_envs_params']
    env_num = EnvConfig['env_num']
    env_list = ParallelEnv(args.env, args.xparl_addr, train_envs_params)

    # env for eval
    eval_env_params = EnvConfig['eval_env_params']

    print("Eval Env Params:", eval_env_params)

    eval_env = LocalEnv(args.env, eval_env_params)

    obs_dim = eval_env.obs_dim
    action_dim = eval_env.action_dim

    print("Obs Dim:", obs_dim, "Action Dim:", action_dim)


    # Initialize model, algorithm, agent, replay_memory
    if args.framework == 'torch':
        CarlaModel, SAC, CarlaAgent = TorchModel, TorchSAC, TorchAgent
    elif args.framework == 'paddle':
        CarlaModel, SAC, CarlaAgent = PaddleModel, PaddleSAC, PaddleAgent
    model = CarlaModel(obs_dim, action_dim)
    print("Model Created")
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    print("Algorithm Initialized")
    agent = CarlaAgent(algorithm)
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

    total_steps = 0
    last_save_steps = 0
    test_flag = 0

    obs_list = env_list.reset()
    print("OLD OBSERVATION LIST:", obs_list)
    while total_steps < args.train_total_steps:
        # Train episode

        # if rpm.size() < WARMUP_STEPS:
        if 5 < 2:
            print("@"*20, "WARMUP STEP", "@"*20)
            action_list = [
                np.random.uniform(-1, 1, size=action_dim)
                for _ in range(env_num)
            ]
        else:
            print("ALREADY WARMED UP")
            action_list = []
            for obs in obs_list:
                action_list.append(agent.sample(obs))
            # action_list = [agent.sample(obs) for obs in obs_list]
            print("Action List Returned In Train:", action_list)


        next_obs_list, reward_list, done_list, info_list, next_obs_rgb_list = env_list.step(
            action_list)

        # Store data in replay memory
        for i in range(env_num):
            rpm.append(obs_list[i], action_list[i], reward_list[i],
                       next_obs_rgb_list[i], done_list[i])

        obs_list = env_list.get_obs()
        total_steps = env_list.total_steps
        # print("NEW OBS LIST:", obs_list)
        # break

        #logger.info('----------- Step 1 ------------')
        # Train agent after collecting sufficient data
        # if rpm.size() >= WARMUP_STEPS:
        if 5 > 2:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            print("BATCH SAMPLED")
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)
        print("-------------------------")

        #logger.info('----------- Step 2 ------------')
        # Save agent
        # if total_steps > int(1e5) and total_steps > last_save_steps + int(1e4):
        if total_steps > int(1) and total_steps > last_save_steps + int(2):
            agent.save('./{model_framework}_model_{train_context}/step_{current_steps}_model.ckpt'.format(
                model_framework=args.framework, current_steps=total_steps, train_context=EnvConfig['train_context']))
            last_save_steps = total_steps
        '''
        #logger.info('----------- Step 3 ------------')
        # Evaluate episode
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            avg_reward = run_evaluate_episodes(agent, eval_env, EVAL_EPISODES)
            tensorboard.add_scalar('eval/episode_reward', avg_reward,
                                   total_steps)
            logger.info(
                'Total steps {}, Evaluation over {} episodes, Average reward: {}'
                .format(total_steps, EVAL_EPISODES, avg_reward))
        '''
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
