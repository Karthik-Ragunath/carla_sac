import numpy as np
from parl.utils import logger

class ReplayMemory(object):
    def __init__(self, max_size, obs_dim, act_dim, merge_images=True, openai_mode=False):
        """ create a replay memory for off-policy RL or offline RL.

        Args:
            max_size (int): max size of replay memory
            obs_dim (list or tuple): observation shape
            act_dim (list or tuple): action shape
        """
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # self.obs = np.zeros((max_size, obs_dim), dtype='float32')
        if not openai_mode:
            if merge_images:
                self.obs = np.zeros((max_size, 2, 300, 300, 3), dtype='float32')
            else:
                self.obs = np.zeros((max_size, 1, 300, 300, 3), dtype='float32')
        else:
            if merge_images:
                self.obs = np.zeros((max_size, 2, 224, 224, 3), dtype='float32')
            else:
                self.obs = np.zeros((max_size, 1, 224, 224, 3), dtype='float32')

        if act_dim == 0:  # Discrete control environment
            self.action = np.zeros((max_size, ), dtype='int32')
        else:  # Continuous control environment
            self.action = np.zeros((max_size, act_dim), dtype='float32')
        self.reward = np.zeros((max_size, ), dtype='float32')
        self.terminal = np.zeros((max_size, ), dtype='bool')
        # self.next_obs = np.zeros((max_size, obs_dim), dtype='float32')
        if not openai_mode:
            if merge_images:
                self.next_obs = np.zeros((max_size, 2, 300, 300, 3), dtype='float32')
            else:
                self.next_obs = np.zeros((max_size, 1, 300, 300, 3), dtype='float32')
        else:
            if merge_images:
                self.next_obs = np.zeros((max_size, 2, 224, 224, 3), dtype='float32')
            else:
                self.next_obs = np.zeros((max_size, 1, 224, 224, 3), dtype='float32')

        self._curr_size = 0
        self._curr_pos = 0

    def sample_batch(self, batch_size):
        """ sample a batch from replay memory

        Args:
            batch_size (int): batch size

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
        batch_idx = np.random.randint(self._curr_size, size=batch_size)

        obs = self.obs[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal

    def sample_sequentially(self, batch_size, sequential_size=20):
        max_size = self.max_size - 1
        current_pos = self._curr_pos
        left_max = current_pos - sequential_size
        if left_max < 0:
            reversed_size = -left_max
            sampled_indices_1 = np.arange(current_pos)[::-1]
            sampled_indices_2 = np.arange(max_size, max_size-reversed_size, -1)
            sampled_indices = np.concatenate((sampled_indices_1, sampled_indices_2))
            rest_samplable_sequence = np.arange(current_pos, max_size-reversed_size)
            random_range = np.random.choice(rest_samplable_sequence, size = batch_size - sequential_size, replace=False)
            sampled_indices = np.concatenate((sampled_indices, random_range))
        else:
            sampled_indices = np.arange(left_max, current_pos)[::-1]
            rest_indices_left = np.arange(0, left_max)
            rest_indices_right = np.arange(current_pos, max_size)
            rest_indices_combined = np.concatenate((rest_indices_left, rest_indices_right))
            random_choice = np.random.choice(rest_indices_combined, size=batch_size - sequential_size, replace=False)
            sampled_indices = np.concatenate((sampled_indices, random_choice))

        obs = self.obs[sampled_indices]
        reward = self.reward[sampled_indices]
        action = self.action[sampled_indices]
        next_obs = self.next_obs[sampled_indices]
        terminal = self.terminal[sampled_indices]

        return obs, action, reward, next_obs, terminal

    def make_index(self, batch_size):
        """ sample a batch of indexes

        Args:
            batch_size (int): batch size

        Returns:
            batch of indexes
        """
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        return batch_idx

    def sample_batch_by_index(self, batch_idx):
        """ sample a batch from replay memory by indexes

        Args:
            batch_idx (list or np.array): batch indexes

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
        obs = self.obs[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal

    def append(self, obs, act, reward, next_obs, terminal):
        """ add an experience sample at the end of replay memory

        Args:
            obs (float32): observation, shape of obs_dim
            act (int32 in Continuous control environment, float32 in Continuous control environment): action, shape of act_dim
            reward (float32): reward
            next_obs (float32): next observation, shape of obs_dim
            terminal (bool): terminal of an episode or not
        """
        if self._curr_size < self.max_size:
            self._curr_size += 1
        self.obs[self._curr_pos] = obs
        self.action[self._curr_pos] = act
        self.reward[self._curr_pos] = reward
        self.next_obs[self._curr_pos] = next_obs
        self.terminal[self._curr_pos] = terminal
        self._curr_pos = (self._curr_pos + 1) % self.max_size

    def size(self):
        """ get current size of replay memory.
        """
        return self._curr_size

    def __len__(self):
        return self._curr_size

    def save(self, pathname):
        """ save replay memory to local file (numpy file format: *.npz).
        """
        other = np.array([self._curr_size, self._curr_pos], dtype=np.int32)
        np.savez(
            pathname,
            obs=self.obs,
            action=self.action,
            reward=self.reward,
            terminal=self.terminal,
            next_obs=self.next_obs,
            other=other)

    def load(self, pathname):
        """ load replay memory from local file (numpy file format: *.npz).
        """
        data = np.load(pathname)
        other = data['other']
        if int(other[0]) > self.max_size:
            logger.warn('loading from a bigger size rpm!')
        self._curr_size = min(int(other[0]), self.max_size)
        self._curr_pos = min(int(other[1]), self.max_size - 1)

        self.obs[:self._curr_size] = data['obs'][:self._curr_size]
        self.action[:self._curr_size] = data['action'][:self._curr_size]
        self.reward[:self._curr_size] = data['reward'][:self._curr_size]
        self.terminal[:self._curr_size] = data['terminal'][:self._curr_size]
        self.next_obs[:self._curr_size] = data['next_obs'][:self._curr_size]
        logger.info("[load rpm]memory loade from {}".format(pathname))

    def load_from_d4rl(self, dataset):
        """ load data from d4rl dataset(https://github.com/rail-berkeley/d4rl#using-d4rl) to replay memory.

        Args:
            dataset(dict): dataset that contains:
                            observations (np.float32): shape of (batch_size, obs_dim),
                            next_observations (np.int32): shape of (batch_size, obs_dim),
                            actions (np.float32): shape of (batch_size, act_dim),
                            rewards (np.float32): shape of (batch_size),
                            terminals (bool): shape of (batch_size)
        
        Example:

        .. code-block:: python

            import gym
            import d4rl

            env = gym.make("hopper-medium-v0")
            rpm = ReplayMemory(max_size=int(2e6), obs_dim=11, act_dim=3)
            rpm.load_from_d4rl(d4rl.qlearning_dataset(env))

            # Output

            # Dataset Info: 
            # key: observations,	shape: (999981, 11),	dtype: float32
            # key: actions,	shape: (999981, 3),	dtype: float32
            # key: next_observations,	shape: (999981, 11),	dtype: float32
            # key: rewards,	shape: (999981,),	dtype: float32
            # key: terminals,	shape: (999981,),	dtype: bool
            # Number of terminals on: 3045

        """
        logger.info("Dataset Info: ")
        for key in dataset:
            logger.info('key: {},\tshape: {},\tdtype: {}'.format(
                key, dataset[key].shape, dataset[key].dtype))
        assert 'observations' in dataset
        assert 'next_observations' in dataset
        assert 'actions' in dataset
        assert 'rewards' in dataset
        assert 'terminals' in dataset

        self.obs = dataset['observations']
        self.next_obs = dataset['next_observations']
        self.action = dataset['actions']
        self.reward = dataset['rewards']
        self.terminal = dataset['terminals']
        self._curr_size = dataset['terminals'].shape[0]
        assert self._curr_size <= self.max_size, 'please set a proper max_size for ReplayMemory'
        logger.info('Number of terminals on: {}'.format(self.terminal.sum()))
