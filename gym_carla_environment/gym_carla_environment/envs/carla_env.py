import carla
import gym
from gym import spaces
import logging
import numpy as np
from queue import Queue
from autopilot.environment.carla import CarlaEnvironment
from autopilot.agent import AutonomousVehicle
import time

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
        self.generate_traffic = params['generate_traffic']
        self.num_traffic_vehicles = params['num_traffic_vehicles']
        self.num_pedestrians = params['num_pedestrians']
        self.queue_length = 0
        return

    def block_msg_queue(self):
        steer, throttle, brake = self.msg_queue.get()
        self.queue_length -= 1
        # LOGGER.info(f"steer: {steer} size: {self.msg_queue.qsize()}")
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
        self.queue_length = 0
        self.location_history = []
        self.zero_speed_stop_count = 0
        if not self.carla_environment:
            self.carla_environment = CarlaEnvironment(
                carla_host=self.params['carla_host'],
                carla_port=self.params['carla_port'],
                carla_timeout=self.params['carla_timeout'],
                sync=self.params['sync'],
                tick_interval=self.params['tick_interval']            
            )
            self.agent_vehicle = self.carla_environment.spawn_agent_vehicle(fixed_spawn=False)
            if self.generate_traffic:
                self.carla_environment.spawn_traffic(
                    number_of_vehicles=self.num_traffic_vehicles,
                    number_of_walkers=self.num_pedestrians
                )
            self.agent_vehicle.autopilot = False
            self.agent_vehicle.control = None
            self.agent_vehicle.set_autopilot(False, stop_vehicle=False)
            # time.sleep(4)
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
            if self.generate_traffic:
                self.carla_environment.spawn_traffic(
                    number_of_vehicles=self.num_traffic_vehicles,
                    number_of_walkers=self.num_pedestrians
                )
            self.agent_vehicle.autopilot = False
            self.agent_vehicle.control = None
            self.agent_vehicle.set_autopilot(False, stop_vehicle=False)
            # time.sleep(4)
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
            self.queue_length += 1
            self.msg_queue.put(action)
            while self.queue_length > 0:
                continue
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
            LOGGER.info('terminating due to crash')
            return -200

        # if self.steer < 0:
        #     right_steer = -(self.steer)
        #     left_steer = 0
        #     # LOGGER.info(f"right steer: {right_steer}")
        # else:
        #     right_steer = 0
        #     left_steer = self.steer
        #     # LOGGER.info(f"left steer: {left_steer}")

        # Alternate way to compute forward
        '''
        yaw_global = np.radians(vehicle.get_transform().rotation)
        rotation_global = np.array([
            [np.sin(yaw_global),  np.cos(yaw_global)],
            [np.cos(yaw_global), -np.sin(yaw_global)]
        ])

        velocity_global = vehicle.get_velocity()
        velocity_global = np.array([velocity_global.y, velocity_global.x])
        velocity_local = rotation_global.T @ velocity_global
        '''

        current_velocity = self.agent_vehicle.get_velocity() # m/s
        curr_velocity_array = np.array([current_velocity.x, current_velocity.y])
        curr_velocity_norm = np.linalg.norm(curr_velocity_array)
        speed_kmph = 3.6 * curr_velocity_norm

        if speed_kmph < 2:
            self.zero_speed_stop_count += 1
        else:
            self.zero_speed_stop_count = 0
        
        # <10 * x> -> x seconds in simualtor world
        # sensor_tick - 0.1 seconds
        # fixed_time_delta_between_frames = 0.025
        # skip_frames = 0.1 / 0.025 = 4
        # 1 step - 4 ticks - 0.1 seconds
        # 60 steps - 0.1 * 60 = 6 seconds in simulator world

        if self.zero_speed_stop_count >= 60:
            self.terminated = True
            LOGGER.error('Terminating due to speed deficit.')
            return -200

        location_obj = self.agent_vehicle.get_transform().location
        distance_reward = 0
        if len(self.location_history) == 10:
            distance_reward = abs(location_obj.distance(self.location_history[0])) + abs(location_obj.distance(self.location_history[4]))
            self.location_history.pop(0)
        elif len(self.location_history) > 0:
            distance_reward = abs(location_obj.distance(self.location_history[0]))
        self.location_history.append(location_obj)
        # reward = speed_kmph / 5 + (left_steer * -0.5) + (right_steer * -0.5) + (self.throttle * 1) + (self.brake * -0.5)
        reward = speed_kmph / 5 + distance_reward

        return reward
    
    def check_terminal(self):
        """Check if collision occured."""
        terminated = False
        if self.agent_vehicle.sensors['collision'].fetch():
            terminated = True
        return terminated
