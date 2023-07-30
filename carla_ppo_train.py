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
import pandas as pd
from pathlib import Path
import cv2
import torch.optim as optim
from tqdm import tqdm

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
parser.add_argument("--load_imitation", action='store_true', help='load from imitation learning model')
parser.add_argument("--imitation_context", type=str, default="params_imitation_1")
parser.add_argument("--imitation_data_dir", type=str, help='directory where imitation data is present',  required=False)
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
    if args.load_imitation:
        pretrained_epoch = agent.load_param_imitation(load_context=args.imitation_context)
    else:
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
    step_loss_index = 0
    for i_ep in range(pretrained_epoch + 1, args.num_episodes):
        score = 0
        # retry = True
        # while retry:
        #     try:
        #         state = env.reset(episode_num=i_ep)
        #         retry = False
        #     except Exception as e:
        #         LOGGER.exception(f"EXCEPTION DURING RESET: - {e}")
        if not args.load_imitation:
            state = env.reset(episode_num=i_ep)
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
                if step_index % 10 == 0:
                    LOGGER.info('Ep {}\tLast score: {:.2f}\tSteps: {}'.format(i_ep, score, step_index))
                if done or die:
                    break
            running_score = running_score * 0.99 + score * 0.01

            if score > best_episode_reward:
                best_episode_reward = score
                agent.save_checkpoint_reward(i_ep)

            if running_score > best_episode_running_score:
                best_episode_running_score = running_score
                agent.save_checkpoint_running_score(i_ep)

            LOGGER.info('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}\t Steps: {}'.format(i_ep, score, running_score, step_index))

            writer.add_scalar('train_reward_score', score, i_ep)
            writer.add_scalar('train_reward_running_score', running_score, i_ep)
            writer.add_scalar('train_steps', step_index, i_ep)

            if i_ep % args.log_interval == 0:
                if args.vis:
                    draw_reward(xdata=i_ep, ydata=running_score)
                print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
                agent.save_param()
            if running_score > args.running_score:
                print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
                break
        else:
            mse_loss = torch.nn.MSELoss()
            optimizer = optim.Adam(agent.net.parameters(), lr=1e-3)
            epoch_loss = 0
            towns = os.listdir(args.imitation_data_dir)
            new_width, new_height = 96, 96
            total_epoch_steps = 0
            for town in towns:
                town_dir = os.path.join(args.imitation_data_dir, town)
                train_runs = os.listdir(town_dir)
                for train_run in train_runs:
                    data_frame_path = os.path.join(town_dir, train_run, 'pd_dataframe.pkl')
                    gt_df = pd.read_pickle(data_frame_path)
                    for row_index in tqdm(range(len(gt_df))):
                        row = gt_df.iloc[row_index]
                        abs_image_path = os.path.join(town_dir, train_run, row['image_path'])
                        state = cv2.imread(abs_image_path)
                        state = cv2.resize(state, (new_width, new_height))
                        state = np.transpose(state, axes=(2, 0, 1))
                        action_predicted = agent.select_action_imitation(state)
                        action_predicted = torch.squeeze(action_predicted, 0)
                        action = row['action']
                        action[0], action[1] = action[1], action[0]
                        action = torch.from_numpy(action).double().to(agent.device)
                        loss = mse_loss(action_predicted, action)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        step_loss = loss.item()
                        epoch_loss += step_loss
                        writer.add_scalar('step_loss', step_loss, step_loss_index)
                        step_loss_index += 1
                        total_epoch_steps += 1
                    logging.info(f"epoch: {i_ep}, town: {town}, run: {train_run} completed")
                os.makedirs("imitation_models", exist_ok=True)
                torch.save(agent.net.state_dict(), os.path.join("imitation_models", f"epoch__{i_ep}__{town}.pt"))
            epoch_loss /= total_epoch_steps
            writer.add_scalar('epoch_loss', epoch_loss, i_ep)
