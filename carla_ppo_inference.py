import argparse
import numpy as np
import torch
from utils_ppo import DrawLine
import logging
from torch.utils.tensorboard import SummaryWriter
from torch_base.torch_agent_inference_ppo import Agent
from ppo_env import Env
from env_config import EnvConfig
import os

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument("--device_id", "-dev", type=int, default=0, required=False)
parser.add_argument("--log_seed", type=int, default=0, required=False)
parser.add_argument("--num_episodes", type=int, default=1, required=False)
parser.add_argument("--context", type=str, default='inference', required=False)
parser.add_argument("--num_steps_per_episode", type=int, default=250, required=False)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{args.device_id}" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

LOGGER= logging.getLogger()
LOGGER.setLevel(logging.DEBUG) # or whatever
handler = logging.FileHandler(f"ppo_logger_inference_{args.log_seed}.log", 'w', 'utf-8')
formatter = logging.Formatter('%(name)s %(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)

if __name__ == "__main__":
    agent = Agent(device=device, context=args.context, args=args)
    agent.load_param(file_dir_path="param")
    env = Env(args=args, env_params=EnvConfig['test_env_params'], context=args.context, device=device)

    training_records = []
    running_score = 0
    for i_ep in range(args.num_episodes):
        score = 0
        state = env.reset(episode_num=i_ep)

        for t in range(args.num_steps_per_episode):
            action = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        LOGGER.info('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))