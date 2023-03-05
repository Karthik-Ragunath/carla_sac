import carla
import gym
from gym import spaces
import logging
import numpy as np
from queue import Queue
from autopilot.environment.carla import CarlaEnvironment
from autopilot.agent import AutonomousVehicle

LOGGER = logging.getLogger(__name__)

class CarlaEnv(gym.Env):
    """OpenAI gym wrapper for CARLA simulator."""
    def __init__(self, params):
        LOGGER.info("Init of CarlaEnv instance called.")
        self.params = params
        self.action_space = spaces.Box(
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-50.0,
            high=50.0,
            shape=(512,),
            dtype=np.float32
        )
        self.carla_environment = None
        self.steer = None
        self.throttle = None
        self.brake = None
        return

    def block_msg_queue(self):
        steer, throttle, brake = self.msg_queue.get()
        vehicle_action = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )
        self.agent_vehicle.vehicle.apply_control(vehicle_action)
        return

    def reset(self):
        """Reset function."""
        self.msg_queue = Queue()
        if not self.carla_environment:
            self.carla_environment = CarlaEnvironment(
                carla_host=self.params['carla_host'],
                carla_port=self.params['carla_port'],
                carla_timeout=self.params['carla_timeout'],
                sync=self.params['sync'],
                tick_interval=self.params['tick_interval']            
            )
            self.agent_vehicle = self.carla_environment.spawn_agent_vehicle(fixed_spawn=False)
            self.agent_vehicle.autopilot = False
            self.agent_vehicle.control = None
            self.agent_vehicle.set_autopilot(False, stop_vehicle=False)
            self.carla_environment.add_tick_callback(self.block_msg_queue)
        else:
            self.carla_environment.close()
            self.carla_environment = CarlaEnvironment(
                carla_host=self.params['carla_host'],
                carla_port=self.params['carla_port'],
                carla_timeout=self.params['carla_timeout'],
                sync=self.params['sync'],
                tick_interval=self.params['tick_interval']            
            )
            self.agent_vehicle = self.carla_environment.spawn_agent_vehicle(fixed_spawn=False)
            self.agent_vehicle.autopilot = False
            self.agent_vehicle.control = None
            self.agent_vehicle.set_autopilot(False, stop_vehicle=False)
            self.carla_environment.add_tick_callback(self.block_msg_queue)
        current_snapshot = self.agent_vehicle.sensors['front_camera'].fetch()
        frame_number = self.carla_environment.frame
        return current_snapshot, {"frame_number": frame_number}

    def step(self, action):
        """Step function."""
        try:
            steer, throttle, brake = action
            self.steer = steer
            self.throttle = throttle
            self.brake = brake
            self.msg_queue.put(action)
            self.reward = self.get_reward()
            frame_number = self.carla_environment.frame
            return self.agent_vehicle.sensors['front_camera'].fetch(), self.reward, self.terminated, False, {"frame_number": frame_number}
        except Exception as e:
            LOGGER.exception(f"exception raised - {e}")
            return None, -500, True, False, {}

    def get_reward(self):
        """Get reward function."""
        self.terminated = self.check_terminal()
        if self.terminated:
            return -500
        
        # current_velocity = self.agent_vehicle.get_velocity() # m/s
        # curr_velocity_array = np.array([current_velocity.x, current_velocity.y])
        # curr_velocity_norm = np.linalg.norm(curr_velocity_array)
        # reward = 3.6 * curr_velocity_norm # kmph

        if self.steer < 0:
            right_steer = -(self.steer)
            left_steer = 0
        else:
            right_steer = 0
            left_steer = self.steer

        current_velocity = self.agent_vehicle.get_velocity() # m/s
        curr_velocity_array = np.array([current_velocity.x, current_velocity.y])
        curr_velocity_norm = np.linalg.norm(curr_velocity_array)
        speed_kmph = 3.6 * curr_velocity_norm
        reward = speed_kmph / 5 + (left_steer * -0.6) + (right_steer * -0.2) + (self.throttle * 1) + (self.brake * -0.4)
        return reward
    
    def check_terminal(self):
        """Check if collision occured."""
        terminated = False
        if self.agent_vehicle.sensors['collision'].fetch():
            terminated = True
        return terminated
