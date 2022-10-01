import parl
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from copy import deepcopy

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

__all__ = ['TorchSAC']
epsilon = 1e-6


class TorchSAC(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 alpha=None,
                 actor_lr=None,
                 critic_lr=None):
        """ SAC algorithm
            Args:
                model(parl.Model): forward network of actor and critic.
                gamma(float): discounted factor for reward computation
                tau (float): decay coefficient when updating the weights of self.target_model with self.model
                alpha (float): Temperature parameter determines the relative importance of the entropy against the reward
                actor_lr (float): learning rate of the actor model
                critic_lr (float): learning rate of the critic model
        """
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(alpha, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.model = model.to(device)
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic_model.parameters(), lr=critic_lr)

    def predict(self, original_image, bounding_box_image):
        act_mean, _ = self.model.policy(original_image, bounding_box_image)
        action = torch.tanh(act_mean)
        return action

    def sample(self, normal_image_obs, bounded_image_obs):
        act_mean, act_log_std = self.model.policy(normal_image_obs, bounded_image_obs)
        normal = Normal(act_mean, act_log_std.exp())
        # for reparameterization trick  (mean + std*N(0,1))
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - action.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdims=True)
        return action, log_prob

    def learn(self, obs, action, reward, next_obs, terminal):
        critic_loss = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        actor_loss = self._actor_learn(obs)

        self.sync_target()
        return critic_loss, actor_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():
            next_rgb_image = next_obs[:, 0, :, :, :]
            next_bounded_rgb_image = next_obs[:, 1, :, :, :]
            next_rgb_image = next_rgb_image.float().permute(0, 3, 1, 2)
            next_bounded_rgb_image = next_bounded_rgb_image.float().permute(0, 3, 1, 2)
            print("Tensor Sizes:", next_rgb_image.size(), next_bounded_rgb_image.size())
            next_action, next_log_pro = self.sample(next_rgb_image, next_bounded_rgb_image)
            # q1_next, q2_next = self.target_model.critic_model(
            #     next_obs, next_action)
            q1_next, q2_next = self.target_model.critic_model(
                next_rgb_image, next_bounded_rgb_image, next_action)
            target_Q = torch.min(q1_next, q2_next) - self.alpha * next_log_pro
            target_Q = reward + self.gamma * (1. - terminal) * target_Q
        rgb_image = obs[:, 0, :, :, :]
        bounded_rgb_image = obs[:, 1, :, :, :]
        rgb_image = rgb_image.float().permute(0, 3, 1, 2)
        bounded_rgb_image = bounded_rgb_image.float().permute(0, 3, 1, 2)
        # cur_q1, cur_q2 = self.model.critic_model(obs, action)
        cur_q1, cur_q2 = self.model.critic_model(rgb_image, bounded_rgb_image, action)
        critic_loss = F.mse_loss(cur_q1, target_Q) + F.mse_loss(
            cur_q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs):
        rgb_image = obs[:, 0, :, :, :]
        bounded_rgb_image = obs[:, 1, :, :, :]
        rgb_image = rgb_image.float().permute(0, 3, 1, 2)
        bounded_rgb_image = bounded_rgb_image.float().permute(0, 3, 1, 2)
        # act, log_pi = self.sample(obs)
        # q1_pi, q2_pi = self.model.critic_model(obs, act)
        act, log_pi = self.sample(rgb_image, bounded_rgb_image)
        q1_pi, q2_pi = self.model.critic_model(rgb_image, bounded_rgb_image, act)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        for param, target_param in zip(self.model.parameters(),
                                       self.target_model.parameters()):
            target_param.data.copy_((1 - decay) * param.data +
                                    decay * target_param.data)
