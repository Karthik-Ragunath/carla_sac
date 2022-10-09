import copy
import carla
import gym
import gym_carla
import numpy as np
from parl.utils import logger, tensorboard
from parl.env.continuous_wrappers import ActionMappingWrapper
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys
import os
import shutil
# sys.path.append('/media/karthikragunath/Personal-Data/carla_sac/')
from torch_base import DetectBoundingBox

class Env(object):
    def __init__(self, env_name, train_envs_params):
        self.env = CarlaEnv(env_name=env_name, params=train_envs_params)
        self.episode_reward = 0
        self.episode_steps = 0
        self._max_episode_steps = train_envs_params['max_time_episode']
        self.obs_dim = self.env.env.observation_space.shape[0]
        self.action_dim = self.env.env.action_space.shape[0]
        self.total_steps = 0
        self.episode_count = 0

    def reset(self):
        while True:
            obs = self.env.reset()
            if not obs[1].any():
                continue
            else:
                break
        obs_tup = (obs[0], obs[1])
        self.obs = np.array(obs_tup)
        return self.obs

    def step(self, action):
        return_tuple = self.env.step(action)
        return_list, numpy_rgb_image, bounding_box_image = return_tuple
        # if numpy_rgb_image.any():
            # print("Image Does Exists")
        self.reward = return_list[0]
        self.done = return_list[1]
        self.next_obs_rgb = np.array((numpy_rgb_image, bounding_box_image))
        return self.reward, self.done, self.next_obs_rgb

    def get_obs(self):
        self.total_steps += 1
        self.episode_steps += 1
        self.episode_reward += self.reward
        self.obs = self.next_obs_rgb
        if self.done or self.episode_steps >= self._max_episode_steps:
            # TODO : Change to save every 15 episode
            if self.episode_count % 15 == 0:
                self.episode_count += 1
                self.env.save_episode = True
                self.env.episode_num = self.episode_count
                dir_path = os.path.join(os.getcwd(), 'image_outputs', str(self.env.episode_num))
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                os.mkdir(dir_path)
            else:
                self.episode_count += 1
                self.env.save_episode = False
                self.env.episode_num = -1
            tensorboard.add_scalar('train/episode_reward',
                                    self.episode_reward,
                                    self.total_steps)
            logger.info('Train env done, Reward: {}'.format(self.episode_reward))

            self.episode_steps = 0
            self.episode_reward = 0
            obs_from_reset = self.env.reset()
            obs = (obs_from_reset[0], obs_from_reset[1])
            self.obs = np.array(obs)
        else:
            # print("EPISODE NOT DONE - CONTINUING")
            pass
        return self.obs


class LocalEnv(object):
    def __init__(self, env_name, params):
        self.env = gym.make(env_name, params=params)
        self.env = ActionMappingWrapper(self.env)
        self._max_episode_steps = int(params['max_time_episode'])
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

    def to_bgra_array(self, image):
        """Convert a CARLA raw image to a BGRA numpy array."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        return array

    def to_rgb_array(self, image):
        """Convert a CARLA raw image to a RGB numpy array."""
        array = self.to_bgra_array(image)
        # Convert BGRA to RGB.
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def reset(self):
        obs, _, current_image = self.env.reset()
        if current_image:
            numpy_rgb_image = self.to_rgb_array(current_image)
            plt.imshow(numpy_rgb_image)
            plt.savefig("carla_rgb_sensor_detected/" + str(current_image.frame) + '.png')
        return obs, current_image

    def step(self, action):
        action_out, current_image = self.env.step(action)
        if current_image:
            numpy_rgb_image = self.to_rgb_array(current_image)
            plt.imshow(numpy_rgb_image)
            plt.savefig("carla_rgb_sensor_detected/" + str(current_image.frame) + '.png')
        return action_out, current_image

class CarlaEnv(object):
    def __init__(self, env_name, params):
        class ActionSpace(object):
            def __init__(self,
                         action_space=None,
                         low=None,
                         high=None,
                         shape=None,
                         n=None):
                self.action_space = action_space
                self.low = low
                self.high = high
                self.shape = shape
                self.n = n

            def sample(self):
                return self.action_space.sample()
        self.env = gym.make(env_name, params=params)
        self._max_episode_steps = int(params['max_time_episode'])
        self.action_space = ActionSpace(
            self.env.action_space, self.env.action_space.low,
            self.env.action_space.high, self.env.action_space.shape)
        self.save_episode = False
        self.episode_num = -1

    def to_bgra_array(self, image):
        """Convert a CARLA raw image to a BGRA numpy array."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        return array

    def to_rgb_array(self, image):
        """Convert a CARLA raw image to a RGB numpy array."""
        array = self.to_bgra_array(image)
        # Convert BGRA to RGB.
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def reset(self):
        current_image = self.env.reset()
        bounded_image = None
        numpy_rgb_image = None
        if current_image:
            numpy_rgb_image = self.to_rgb_array(current_image)
            faster_rcnn_obj = DetectBoundingBox(numpy_rgb_image, str(current_image.frame) + '.png')
            bounded_image = faster_rcnn_obj.detect_bounding_boxes()
            if self.save_episode:
                fig = plt.figure()
                plt.imshow(bounded_image)
                plt.savefig(os.path.join(os.getcwd(), 'image_outputs', str(self.episode_num), (str(current_image.frame) + '.png')))
                plt.close(fig)
        else:
            print("NO IMAGE DETECTED FOR NOW IN RESET")
        return numpy_rgb_image, bounded_image

    def step(self, action):
        assert np.all(((action<=1.0 + 1e-3), (action>=-1.0 - 1e-3))), \
            'the action should be in range [-1.0, 1.0]'
        mapped_action = self.action_space.low + (action - (-1.0)) * (
            (self.action_space.high - self.action_space.low) / 2.0)
        mapped_action = np.clip(mapped_action, self.action_space.low, self.action_space.high)
        action_out, current_image = self.env.step(mapped_action)
        bounded_image = None
        numpy_rgb_image = None
        if current_image:
            numpy_rgb_image = self.to_rgb_array(current_image)
            faster_rcnn_obj = DetectBoundingBox(numpy_rgb_image, str(current_image.frame) + '.png')
            bounded_image = faster_rcnn_obj.detect_bounding_boxes()
            if self.save_episode:
                fig = plt.figure()
                plt.imshow(bounded_image)
                plt.savefig(os.path.join(os.getcwd(), 'image_outputs', str(self.episode_num), (str(current_image.frame) + '.png')))
                plt.close(fig)
        else:
            print("NO IMAGE DETECTED FOR NOW IN STEP")
        return action_out, numpy_rgb_image, bounded_image
