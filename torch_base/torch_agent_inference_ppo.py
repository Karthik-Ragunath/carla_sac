from .torch_model_ppo import Net
import torch
import numpy as np
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
from typing import Tuple
import glob
import os
from pathlib import Path

class Agent():
    """
    Agent for testing
    """
    def __init__(self, device, context, args):
        self.args = args
        self.device = device
        self.net = Net(self.args.img_stack).float().to(self.device)
        self.context = context

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)
        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self, file_dir_path):
        filenames = glob.glob(os.path.join(file_dir_path, "reward_checkpoint_*.pkl"))
        model_filename = None
        max_train_epoch = 0
        for filename in filenames:
            complete_path = filename 
            filename = Path(filename).stem
            epoch_num = int(filename.split('_')[-1])
            if epoch_num > max_train_epoch:
                max_train_epoch = epoch_num
                model_filename = complete_path
                pretrained_steps = max_train_epoch
        if model_filename:
            self.net.load_state_dict(torch.load(
                os.path.join(os.getcwd(), model_filename), map_location=self.device
            ))

class ImitationLearningAgent():
    """
    Agent for ImitationLearning
    """
    def __init__(self, device, env_params, context, args):
        self.args = args
        self.env_params = env_params
        self.device = device
        self.net = Net(self.args.img_stack).float().to(self.device)
        self.context = context
        self.net.train()

    def save_param(self):
        torch.save(self.net.state_dict(), os.path.join(self.checkpoints_save_dir, 'ppo_net_imitation_model_trained.pkl'))

    def select_action(self, state: torch.Tensor):
        # state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        # state = state.to(self.device).unsqueeze(0)
        state = state.to(self.device)
        alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)
        # action = action.squeeze().cpu().numpy()
        return action

    def load_param(self, load_context):
        if load_context:
            checkpoints_save_dir = 'params_' + load_context
        else:
            checkpoints_save_dir = self.checkpoints_save_dir
        max_train_epoch = -1
        if self.env_params.get('load_recent_model', False):
            filenames = glob.glob(os.path.join(checkpoints_save_dir, "reward_checkpoint*.pkl"))
            model_filename = None
            for filename in filenames:
                complete_path = filename 
                filename = Path(filename).stem
                epoch_num = int(filename.split('_')[-1])
                if epoch_num > max_train_epoch:
                    max_train_epoch = epoch_num
                    model_filename = complete_path
            if model_filename:
                self.net.load_state_dict(torch.load(
                    os.path.join(os.getcwd(), model_filename), map_location=self.device
                ))
        return max_train_epoch