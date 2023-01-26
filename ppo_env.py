import gym
import numpy as np
from torch_base.detect_bounding_boxes import DetectBoundingBox
import matplotlib.pyplot as plt
import os
# from skimage.transform import resize
import logging

# Necessary to create custom gym environments
import gym_carla


LOGGER = logging.getLogger(__name__)

class Env(object):
    """
    Environment wrapper for CarRacing 
    """
    def __init__(self, args, env_params, train_context_name):
        self.args = args
        self.env = CarlaEnv(env_name='carla-v0', params=env_params, context=train_context_name)
        self.obs_dim = self.env.env.observation_space.shape[0]
        self.action_dim = self.env.env.action_space.shape[0]

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        # TODO: Hard-coding zero index for retrieving actual RGB image alone and not bounding box image
        # TODO: Make it config driven to retrieve both images
        img_rgb = self.env.reset()[0]
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.args.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(self.args.action_repeat):
            img_rgb_tuple, reward, die, _, _ = self.env.step(action)
            img_rgb = img_rgb_tuple[0]
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.args.img_stack
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

class CarlaEnv(object):
    def __init__(self, env_name, params, context):
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
        self.params = params
        self._max_episode_steps = int(params['max_time_episode'])
        self.action_space = ActionSpace(
            self.env.action_space, self.env.action_space.low,
            self.env.action_space.high, self.env.action_space.shape)
        self.save_episode = False
        self.episode_num = -1
        self.eval_episode_num = 0
        self.train_vis_dir = context + '_train'
        self.valid_vis_dir = context + '_valid'

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
                plt.savefig(os.path.join(os.getcwd(), self.train_vis_dir, str(self.episode_num), (str(current_image.frame) + '.png')))
                plt.close(fig)
        else:
            LOGGER.error("NO IMAGE DETECTED FOR NOW IN RESET")
        return numpy_rgb_image, bounded_image

    '''
    def step(self, action, is_validation=False):
        assert np.all(((action<=1.0 + 1e-3), (action>=-1.0 - 1e-3))), \
            'the action should be in range [-1.0, 1.0]'
        # mapped_action = [-2, -2] * ((action + 1) * 2) = action * 2
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
            if self.save_episode or is_validation:
                fig = plt.figure()
                plt.imshow(bounded_image)
                if is_validation:
                    plt.savefig(os.path.join(os.getcwd(), self.valid_vis_dir, str(self.eval_episode_num), (str(current_image.frame) + '.png')))
                else:
                    plt.savefig(os.path.join(os.getcwd(), self.train_vis_dir, str(self.episode_num), (str(current_image.frame) + '.png')))
                plt.close(fig)
        else:
            print("NO IMAGE DETECTED FOR NOW IN STEP")
        return (action_out, numpy_rgb_image, bounded_image)
    '''

    def step(self, action, is_validation=False):
        mapped_action = np.clip(action, self.action_space.low, self.action_space.high)
        current_image, reward, die, _, _ = self.env.step(mapped_action)
        bounded_image = None
        numpy_rgb_image = None
        if current_image:
            numpy_rgb_image = self.to_rgb_array(current_image)
            faster_rcnn_obj = DetectBoundingBox(numpy_rgb_image, str(current_image.frame) + '.png')
            bounded_image = faster_rcnn_obj.detect_bounding_boxes()
            if self.save_episode or is_validation:
                fig = plt.figure()
                plt.imshow(bounded_image)
                if is_validation:
                    plt.savefig(os.path.join(os.getcwd(), self.valid_vis_dir, str(self.eval_episode_num), (str(current_image.frame) + '.png')))
                else:
                    plt.savefig(os.path.join(os.getcwd(), self.train_vis_dir, str(self.episode_num), (str(current_image.frame) + '.png')))
                plt.close(fig)
        else:
            LOGGER.error("NO IMAGE DETECTED FOR NOW IN STEP")
        return (numpy_rgb_image, bounded_image), reward, die, _, _
    
    def render(self, *arg):
        self.env.render()