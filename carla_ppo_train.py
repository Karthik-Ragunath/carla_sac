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
parser.add_argument("--log_seed", type=str, default="0", required=False)
parser.add_argument("--running_score", type=int, default=12000, required=False)
parser.add_argument("--context", type=str, default='train', required=False)
parser.add_argument("--num_episodes", type=int, default=100000, required=False)
parser.add_argument("--num_steps_per_episode", type=int, default=250, required=False)
parser.add_argument("--load_context", type=str, required=False)
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
    env = Env(args=args, env_params=EnvConfig['train_env_params'], context=args.context, device=device)
    agent = Agent(device=device, env_params=EnvConfig['train_env_params'], context=args.context, args=args)
    pretrained_epoch = agent.load_param(load_context=args.load_context)
    if args.vis:
        draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")
    training_records = []
    running_score = 0
    best_episode_reward = 0
    best_episode_running_score = 0
    LOGGER.info("start training")
    checkpoints_save_dir = 'params_' + args.context
    if not os.path.exists(checkpoints_save_dir):
        os.makedirs(checkpoints_save_dir)
    for i_ep in range(pretrained_epoch + 1, args.num_episodes):
        score = 0
        retry = True
        while retry:
            try:
                state = env.reset(episode_num=i_ep)
                retry = False
            except Exception as e:
                LOGGER.exception(f"EXCEPTION DURING RESET: - {e}")

        for step_index in range(args.num_steps_per_episode):
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
        writer.add_scalar('train_steps', step_index, i_ep)

        if i_ep % args.log_interval == 0:
            if args.vis:
                draw_reward(xdata=i_ep, ydata=running_score)
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}\tStep Index: {:.2f}'.format(i_ep, score, running_score, step_index))
            agent.save_param()
        if running_score > args.running_score:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break