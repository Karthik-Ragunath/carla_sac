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
sys.path.append('/media/karthikragunath/Personal-Data/carla_6/RL_CARLA/')
from torch_base import DetectBoundingBox

class ParallelEnv(object):
    def __init__(self, env_name, train_envs_params):
        print("trying to connect to remote env")
        self.env = CarlaEnv(env_name=env_name, params=train_envs_params)
        self.episode_reward = 0
        self.episode_steps = 0
        self._max_episode_steps = train_envs_params['max_time_episode']
        self.total_steps = 0
        print("Init Successfully executed")

    def reset(self):
        while True:
            obs = self.env.reset()
            if not obs[1].any():
                continue
            else:
                break
        obs_tup = (obs[1], obs[2])
        self.obs_tup = np.array(obs_tup)
        return self.obs_tup

    def step(self, action):
        return_tuple = self.env.step(action)
        return_list, numpy_rgb_image, bounding_box_image = return_tuple
        if numpy_rgb_image.any():
            print("Image Does Exists")
        self.next_waypoint_obs = return_list[0]
        self.reward = return_list[1]
        self.done = return_list[2]
        self.info = return_list[3]
        self.next_obs_rgb = (numpy_rgb_image, bounding_box_image)
        # self.next_waypoint_obs = np.array(self.next_waypoint_obs, dtype=object)
        # self.reward = np.array(self.reward, dtype=object)
        # self.done = np.array(self.done, dtype=object)
        # self.info = np.array(self.info, dtype=object)
        # self.next_obs_rgb = np.array(self.next_obs_rgb, dtype=object)
        return self.next_waypoint_obs, self.reward, self.done, self.info, self.next_obs_rgb

    def get_obs(self):
        for i in range(self.env_num):
            self.total_steps += 1
            self.episode_steps_list[i] += 1
            self.episode_reward_list[i] += self.reward_list[i]

            # self.obs_list[i] = self.next_obs_list[i]
            self.obs_list[i] = self.next_obs_rgb_list[i]
            print("INSIDE GET OBS")
            if self.done_list[i] or self.episode_steps_list[
                    i] >= self._max_episode_steps:
                print("EPISODE DONE IN TRAIN")
                tensorboard.add_scalar('train/episode_reward_env{}'.format(i),
                                       self.episode_reward_list[i],
                                       self.total_steps)
                logger.info('Train env {} done, Reward: {}'.format(
                    i, self.episode_reward_list[i]))

                self.episode_steps_list[i] = 0
                self.episode_reward_list[i] = 0
                obs_list_i = self.env_list[i].reset()
                # self.obs_list[i] = obs_list_i.get()
                get_obs = obs_list_i.get()
                obs = (get_obs[1], get_obs[2])
                self.obs_list[i] = np.array(obs)
            else:
                print("EPISODE NOT DONE - CONTINUING")
        return self.obs_list


class LocalEnv(object):
    def __init__(self, env_name, params):
        print("-"*30, "Inside Local Env", "-"*30)
        self.env = gym.make(env_name, params=params)
        print("*" * 40, "Environment Created", "*" * 40)
        self.env = ActionMappingWrapper(self.env)
        # print("Low Bound:", self.env.low_bound)
        # print("High Bound:", self.env.high_bound)
        self._max_episode_steps = int(params['max_time_episode'])
        # print("Max Episodes:", self._max_episode_steps)
        self.obs_dim = self.env.observation_space.shape[0]
        # print('State Space:', self.env.observation_space)
        # print("Obs Dim:", self.obs_dim)
        self.action_dim = self.env.action_space.shape[0]
        # print('Action Space:', self.env.action_space)
        # print("Obs Dim:", self.action_dim)

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
        # print("env_utils.py reset")
        obs, _, current_image = self.env.reset()
        if current_image:
            numpy_rgb_image = self.to_rgb_array(current_image)
            plt.imshow(numpy_rgb_image)
            plt.savefig("carla_rgb_sensor_detected/" + str(current_image.frame) + '.png')
        #     print("$" * 25, "RESET Image Name:", str(current_image.frame), "$" * 25)
        #     faster_rcnn_obj = DetectBoundingBox(numpy_rgb_image, str(current_image.frame) + '.png')
        #     faster_rcnn_obj.detect_bounding_boxes()
        return obs, current_image

    def step(self, action):
        action_out, current_image = self.env.step(action)
        if current_image:
            numpy_rgb_image = self.to_rgb_array(current_image)
            plt.imshow(numpy_rgb_image)
            plt.savefig("carla_rgb_sensor_detected/" + str(current_image.frame) + '.png')
            # print("$" * 25, "STEP Image Name:", str(current_image.frame), "$" * 25)
            # faster_rcnn_obj = DetectBoundingBox(numpy_rgb_image, str(current_image.frame) + '.png')
            # faster_rcnn_obj.detect_bounding_boxes()
        return action_out, current_image

class CarlaEnv(object):
    def __init__(self, env_name, params):
        print("Came Inside Remote Init")
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
        print("Trying To Create Remote GYM Env")
        self.env = gym.make(env_name, params=params)
        print("Remote Env Made")
        self._max_episode_steps = int(params['max_time_episode'])
        self.action_space = ActionSpace(
            self.env.action_space, self.env.action_space.low,
            self.env.action_space.high, self.env.action_space.shape)

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
        print("Parallel ENV reset called")
        obs, _, current_image = self.env.reset()
        print("RESETTING DONE")
        bounded_image = None
        numpy_rgb_image = None
        if current_image:
            numpy_rgb_image = self.to_rgb_array(current_image)
            plt.imshow(numpy_rgb_image)
            plt.savefig("/media/karthikragunath/Personal-Data/carla_6/RL_CARLA/carla_rgb_sensor_flow_detected/" + str(current_image.frame) + '.png')
            print("$" * 25, "RESET Image Name:", str(current_image.frame), "$" * 25)
            faster_rcnn_obj = DetectBoundingBox(numpy_rgb_image, str(current_image.frame) + '.png')
            bounded_image = faster_rcnn_obj.detect_bounding_boxes()
            plt.imshow(bounded_image)
            plt.savefig("/media/karthikragunath/Personal-Data/carla_6/RL_CARLA/carla_rgb_sensor_detected/" + str(current_image.frame) + '.png')
            print("Image Received In ParallelEnv Reset")
        else:
            print("NO IMAGE DETECTED FOR NOW IN RESET")
        return obs, numpy_rgb_image, bounded_image
        # obs, _ = self.env.reset()
        # return obs

    def step(self, action):
        assert np.all(((action<=1.0 + 1e-3), (action>=-1.0 - 1e-3))), \
            'the action should be in range [-1.0, 1.0]'
        mapped_action = self.action_space.low + (action - (-1.0)) * (
            (self.action_space.high - self.action_space.low) / 2.0)
        mapped_action = np.clip(mapped_action, self.action_space.low, self.action_space.high)
        action_out, current_image = self.env.step(mapped_action)
        bounded_image = None
        numpy_rgb_image = None
        print("STEP FUNCTION IN CARLA_ENV_REMOTE CLASS")
        if current_image:
            numpy_rgb_image = self.to_rgb_array(current_image)
            # plt.imshow(numpy_rgb_image)
            # plt.savefig("carla_rgb_sensor_detected/" + str(current_image.frame) + '.png')
            # # print("$" * 25, "STEP Image Name:", str(current_image.frame), "$" * 25)
            faster_rcnn_obj = DetectBoundingBox(numpy_rgb_image, str(current_image.frame) + '.png')
            bounded_image = faster_rcnn_obj.detect_bounding_boxes()
            print("Image Received In ParallelEnv Step")
        else:
            print("NO IMAGE DETECTED FOR NOW IN STEP")
        # return self.env.step(action)
        return action_out, numpy_rgb_image, bounded_image
