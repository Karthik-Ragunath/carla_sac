import argparse
import numpy as np
import torch
from utils_ppo import DrawLine
import logging
from torch.utils.tensorboard import SummaryWriter
from torch_base.torch_agent_ppo import Agent
from ppo_env import Env
from env_config import EnvConfig
import os

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument("--device_id", "-dev", type=int, default=0, required=False)
parser.add_argument("--log_seed", type=int, default=0, required=False)
parser.add_argument("--checkpoints_save_dir", type=str, default="param", required=False)
parser.add_argument("--running_score", type=int, default=2000, required=False)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{args.device_id}" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

LOGGER= logging.getLogger()
LOGGER.setLevel(logging.DEBUG) # or whatever
handler = logging.FileHandler(f"ppo_logger_{args.log_seed}.log", 'w', 'utf-8')
formatter = logging.Formatter('%(name)s %(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)

if __name__ == '__main__':
    writer = SummaryWriter()
    agent = Agent(device=device, args=args)
    env = Env(args=args, env_params=EnvConfig['train_env_params'], train_context_name=EnvConfig['train_context'], device=device)
    if args.vis:
        draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")

    training_records = []
    running_score = 0
    state = env.reset()
    best_episode_reward = 0
    best_episode_running_score = 0
    LOGGER.info("start training")
    # Remove hardcoding directory where models are stored.
    if not os.path.exists(args.checkpoints_save_dir):
        os.makedirs(args.checkpoints_save_dir)
    for i_ep in range(200000):
        score = 0
        state = env.reset()

        for t in range(1000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01

        if score > best_episode_reward:
            best_episode_reward = score
            agent.save_checkpoint_reward(i_ep)

        if running_score > best_episode_running_score:
            best_episode_running_score = running_score
            agent.save_checkpoint_running_score(i_ep)

        LOGGER.info('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))

        writer.add_scalar('train_reward_score', score, i_ep)
        writer.add_scalar('train_reward_running_score', running_score, i_ep)

        if i_ep % args.log_interval == 0:
            if args.vis:
                draw_reward(xdata=i_ep, ydata=running_score)
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            agent.save_param()
        if running_score > args.running_score:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break