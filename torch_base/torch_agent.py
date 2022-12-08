import parl
import torch
import numpy as np

__all__ = ['TorchAgent']


class TorchAgent(parl.Agent):
    def __init__(self, algorithm):
        super(TorchAgent, self).__init__(algorithm)
        # print("TorchAgent Called")
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.alg.sync_target(decay=0)

    def predict(self, obs):
        if self.alg.merge_layer:
            normal_image_obs, bounding_box_image_obs = obs
            normal_image_obs = torch.unsqueeze(torch.from_numpy(normal_image_obs).float(), dim=0).permute(0, 3, 1, 2)
            bounding_box_image_obs = torch.unsqueeze(torch.from_numpy(bounding_box_image_obs).float(), dim=0).permute(0, 3, 1, 2)
        else:
            normal_image_obs = obs
            normal_image_obs = torch.unsqueeze(torch.from_numpy(normal_image_obs).float(), dim=0).permute(0, 3, 1, 2)
            bounding_box_image_obs = None
        if self.alg.merge_layer:
            action = self.alg.predict(normal_image_obs.to(self.device), bounding_box_image_obs.to(self.device))
        else:
            action = self.alg.predict(normal_image_obs.to(self.device))
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy

    def sample(self, obs):
        if self.alg.merge_layer:
            normal_image_obs, bounding_box_image_obs = obs
            normal_image_obs = torch.unsqueeze(torch.from_numpy(normal_image_obs).float(), dim=0).permute(0, 3, 1, 2)
            bounding_box_image_obs = torch.unsqueeze(torch.from_numpy(bounding_box_image_obs).float(), dim=0).permute(0, 3, 1, 2)
        else:
            normal_image_obs = obs
            normal_image_obs = torch.unsqueeze(torch.from_numpy(normal_image_obs).float(), dim=0).permute(0, 3, 1, 2)
            bounding_box_image_obs = None
        if self.alg.merge_layer:
            action, _ = self.alg.sample(normal_image_obs.to(self.device), bounding_box_image_obs.to(self.device))
        else:
            action, _ = self.alg.sample(normal_image_obs.to(self.device))
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy

    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        terminal = torch.FloatTensor(terminal).to(self.device)
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,
                                                 terminal)
        return critic_loss, actor_loss
