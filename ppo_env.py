import gym
import numpy as np
from torch_base.detect_bounding_boxes import DetectBoundingBox
import matplotlib.pyplot as plt
import os
# from skimage.transform import resize
import logging
# Necessary to create custom gym environments
import gym_carla_environment

LOGGER = logging.getLogger(__name__)

class Env(object):
    """
    Environment wrapper for CarRacing 
    """
    def __init__(self, args, env_params, context, device):
        self.args = args
        self.device = device
        if env_params.get('code_mode', 'train') == 'test':
            self.is_inference = True
        else:
            self.is_inference = False
        self.env = CarlaEnv(
            env_name='carla-environment-v0', 
            params=env_params, 
            context=context, 
            device=self.device,
            save_episode=self.is_inference
        )
        self.obs_dim = self.env.env.observation_space.shape[0]
        self.action_dim = self.env.env.action_space.shape[0]

    def reset(self, episode_num: int):
        if episode_num % 50 == 0:
            self.env.save_episode = True
            self.env.episode_num = episode_num
        else:
            self.env.save_episode = False
            self.env.episode_num = episode_num
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
    def __init__(self, env_name, params, context, device, save_episode):
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
        
        self.params = params
        self.env = gym.make(
            env_name,
            params=dict(
                carla_host='127.0.0.1',
                carla_port=self.params['port'],
                carla_timeout=30.0,
                sync=True,
                tick_interval=self.params['sensor_tick'],
                generate_traffic=self.params['generate_traffic'],
                num_traffic_vehicles=self.params['num_traffic_vehicles'],
                num_pedestrians=self.params['num_pedestrians']
            )
        )
        self._max_episode_steps = int(self.params['max_time_episode'])
        self.action_space = ActionSpace(
            self.env.action_space, self.env.action_space.low,
            self.env.action_space.high, self.env.action_space.shape)
        self.save_episode = save_episode
        self.episode_num = -1
        self.eval_episode_num = 0
        self.vis_dir = 'visualization_' + context
        self.device = device
        self.faster_rcnn_obj = DetectBoundingBox(device=self.device)

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
        current_image, aux_dict = self.env.reset()
        frame_number = aux_dict['frame_number']
        bounded_image = None
        numpy_rgb_image = None
        if not current_image is None:
            # numpy_rgb_image = self.to_rgb_array(current_image)
            numpy_rgb_image = current_image
            bounded_image = self.faster_rcnn_obj.detect_bounding_boxes(numpy_rgb_image, str(frame_number) + '.png')
            if self.save_episode:
                fig = plt.figure()
                plt.imshow(bounded_image)
                save_dir = os.path.join(os.getcwd(), self.vis_dir, str(self.episode_num))
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, (str(frame_number) + '.png')))
                plt.close(fig)
        else:
            LOGGER.error("NO IMAGE DETECTED FOR NOW IN RESET")
        return numpy_rgb_image, bounded_image

    def step(self, action):
        mapped_action = np.clip(action, self.action_space.low, self.action_space.high)
        current_image, reward, die, _, aux_dict = self.env.step(mapped_action)
        bounded_image = None
        numpy_rgb_image = None
        if not current_image is None:
            # numpy_rgb_image = self.to_rgb_array(current_image)
            numpy_rgb_image = current_image
            bounded_image = self.faster_rcnn_obj.detect_bounding_boxes(numpy_rgb_image, str(aux_dict['frame_number']) + '.png')
            if self.save_episode:
                fig = plt.figure()
                plt.imshow(bounded_image)
                save_dir = os.path.join(os.getcwd(), self.vis_dir, str(self.episode_num))
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, (str(aux_dict['frame_number']) + '.png')))
                plt.close(fig)
        else:
            LOGGER.error("NO IMAGE DETECTED FOR NOW IN STEP")
        return (numpy_rgb_image, bounded_image), reward, die, _, _
    
    def render(self, *arg):
        self.env.render()