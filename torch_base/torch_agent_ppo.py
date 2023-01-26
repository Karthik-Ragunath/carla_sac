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

class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self, device, args):
        self.args = args
        self.device = device
        self.transition_type = np.dtype(
            [
                ('s', np.float64, 
                (self.args.img_stack, 96, 96)), 
                ('a', np.float64, (3,)), 
                ('a_logp', np.float64),
                ('r', np.float64), ('s_', np.float64, (self.args.img_stack, 96, 96))
            ]
        )
        self.training_step = 0
        self.net = Net(args.img_stack).double().to(self.device)
        self.buffer = np.empty(self.buffer_capacity, dtype=self.transition_type)
        self.counter = 0
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def load_model(self, env_params: dict, file_dir_path: str) -> Tuple[str, int]:
        if env_params.get('load_recent_model', False):
            # set the computation device
            filenames = glob.glob(os.path.join(file_dir_path, "*.pkl"))
            model_filename = None
            max_train_epoch = 0
            for filename in filenames:
                epoch_num = int(filename.split('/')[-1].split('_')[1])
                if epoch_num > max_train_epoch:
                    max_train_epoch = epoch_num
                    model_filename = filename
                    pretrained_steps = max_train_epoch
            if model_filename:
                self.net.load_state_dict(torch.load(
                    os.path.join(os.getcwd(), model_filename), map_location=self.device
                ))
        pass

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self):
        torch.save(self.net.state_dict(), 'param/ppo_net_params_model_trained.pkl')
    
    def save_checkpoint_reward(self, episode):
        torch.save(self.net.state_dict(), f"param/reward_checkpoint_{episode}.pkl")

    def save_checkpoint_running_score(self, episode):
        torch.save(self.net.state_dict(), f"param/run_score_checkpoint_{episode}.pkl")

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = r + self.args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()