#!/usr/bin/env python

# This file is modified by Dongjie yu (yudongjie.moon@foxmail.com)
# from <https://github.com/cjy1992/gym-carla>:
# Copyright (c) 2019:
# author: Jianyu Chen (jianyuchen@berkeley.edu)

from __future__ import division
import copy
import numpy as np
import random
import time
from collections import deque

import gym
from gym import spaces
from gym.utils import seeding
import carla
import cv2

from .coordinates import train_coordinates
from .misc import _vec_decompose, delta_angle_between
from .carla_logger import *
import time

class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params):
        self.logger = setup_carla_logger(
            "output_logger", experiment_name=str(params['port']))
        self.logger.info("Env running in port {}".format(params['port']))
        # parameters
        self.dt = params['dt']
        self.port = params['port']
        self.task_mode = params['task_mode']
        self.code_mode = params['code_mode']
        self.max_time_episode = params['max_time_episode']
        self.obs_size = params['obs_size']
        self.state_size = (self.obs_size[0], self.obs_size[1] - 36)

        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        # TODO : Reset Traffic
        '''
        self.max_traffic_vehicles = params['max_traffic_vehicles']
        '''

        # action and observation space
        # self.action_space = spaces.Box(
        #     np.array([-2.0, -2.0]), np.array([2.0, 2.0]), dtype=np.float32)
        self.action_space = spaces.Box(
            np.array([-1.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-50.0, high=50.0, shape=(512,), dtype=np.float32)

        # Connect to carla server and get world object
        # self._make_carla_client('localhost', self.port)
        self._make_carla_client('127.0.0.1', self.port)
        # Load routes
        self.starts, self.dests = train_coordinates(self.task_mode)
        self.route_deterministic_id = 0

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(
            params['ego_vehicle_filter'], color='49,8,8')

        # TODO : Reset Traffic
        '''
        # Create traffic vehicles (other than ego vehicles)
        self.traffic_bp = self._create_vehicle_bluepprint(
            params['traffic_vehicle_filter'], color='0,0,0')
        '''

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find(
            'sensor.other.collision')
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        if 'image_dimensions' in params:
            capture_width = str(params['image_dimensions'][0])
            capture_height = str(params['image_dimensions'][1])
        else:
            capture_width, capture_height = '300', '300'
        self.camera_bp.set_attribute('image_size_x', capture_width)
        self.camera_bp.set_attribute('image_size_y', capture_height)
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.1')
        # Provide the position of the sensor relative to the vehicle.

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # A list stores the ids for each episode
        self.actors = []

        # store current image
        self.current_image = None

    def compute_side_walk_opp_lane_infractions(self, bounding_box_coordinates):
        in_road_percentage = 0
        in_driving_lane_percentage = 0
        for box_coordinate in bounding_box_coordinates:
            is_in_road = self.map.get_waypoint(location=box_coordinate, project_to_road=False)
            is_in_driving_lane = self.map.get_waypoint(location=box_coordinate, project_to_road=False, lane_type=carla.LaneType.Any)
            if is_in_road:
                in_road_percentage += 1
            if is_in_driving_lane:
                in_driving_lane_percentage += 1
        in_road_percentage = in_road_percentage / 8
        in_driving_lane_percentage = in_driving_lane_percentage / 8

        off_road_percentage = 1 - in_road_percentage
        off_lane_percentage = 1 - in_driving_lane_percentage

        return off_road_percentage, off_lane_percentage

    def reset(self):
        # print("carla_env.py reset function called")
        while True:
            try:
                self.collision_sensor = None
                self.lane_sensor = None

                # Delete sensors, vehicles and walkers
                while self.actors:
                    (self.actors.pop()).destroy()

                # Disable sync mode
                self._set_synchronous_mode(False)

                # Spawn the ego vehicle at a random position between start and dest
                # Start and Destination
                if self.task_mode == 'Straight':
                    self.route_id = 0
                elif self.task_mode == 'Curve':
                    self.route_id = 1  #np.random.randint(2, 4)
                elif self.task_mode == 'Long' or self.task_mode == 'Lane' or self.task_mode == 'Lane_test':
                    if self.code_mode == 'train':
                        # self.route_id = np.random.randint(0, 4)
                        self.route_id = 1
                    elif self.code_mode == 'test':
                        # self.route_id = self.route_deterministic_id
                        # self.route_deterministic_id = (
                        #     self.route_deterministic_id + 1) % 4
                        self.route_id = 1
                elif self.task_mode == 'U_curve':
                    self.route_id = 0
                self.start = self.starts[self.route_id]
                self.dest = self.dests[self.route_id]

                # The tuple (x,y) for the current waypoint
                self.current_wpt = np.array((self.start[0], self.start[1],
                                             self.start[5]))

                ego_spawn_times = 0
                while True:
                    if ego_spawn_times > self.max_ego_spawn_times:
                        self.reset()
                    transform = self._set_carla_transform(self.start)
                    # Code_mode == train, spawn randomly between start and destination
                    '''
                    if self.code_mode == 'train':
                        transform = self._get_random_position_between(
                            start=self.start,
                            dest=self.dest,
                            transform=transform)
                    '''
                    if self._try_spawn_ego_vehicle_at(transform):
                        # TODO : Reset Traffic
                        # code component to spawn traffic vehicles
                        '''
                        # traffic_vehicles_spawned_index = 0
                        # while traffic_vehicles_spawned_index < self.max_traffic_vehicles:
                        #     transform_traffic = self._get_random_position_between(
                        #         start=self.start,
                        #         dest=self.dest,
                        #         transform=transform
                        #     )
                        #     if self._try_spawn_vehicle_at(transform_traffic):
                        #         traffic_vehicles_spawned_index += 1
                        '''
                        break
                    else:
                        ego_spawn_times += 1
                        time.sleep(0.1)

                # Add collision sensor
                self.collision_sensor = self.world.try_spawn_actor(
                    self.collision_bp, carla.Transform(), attach_to=self.ego)
                self.actors.append(self.collision_sensor)
                self.collision_sensor.listen(
                    lambda event: get_collision_hist(event))

                camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
                self.camera_sensor = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.ego)
                self.actors.append(self.camera_sensor)
                self.camera_sensor.listen(lambda data: get_camera_rgb_images(data))


                def get_collision_hist(event):
                    impulse = event.normal_impulse
                    intensity = np.sqrt(impulse.x**2 + impulse.y**2 +
                                        impulse.z**2)
                    self.collision_hist.append(intensity)
                    if len(self.collision_hist) > self.collision_hist_l:
                        self.collision_hist.pop(0)

                def get_camera_rgb_images(data):
                    # image_width = data.width
                    # image_height = data.height
                    # image_transform = data.transform
                    # field_of_view = data.fov
                    # raw_data = data.raw_data
                    # frame_id = data.frame
                    # image_outputs_dir = os.path.join(os.getcwd(), "image_outputs")
                    # data.save_to_disk(image_outputs_dir + '/%.6d.jpg' % data.frame)
                    self.current_image = data

                '''
                measurements, sensor_data = self.client.read_data()
                intersection_otherlane = measurements.player_measurements.intersection_otherlane
                intersection_offroad = measurements.player_measurements.intersection_offroad
                print("*"*30, "Intersection Other Lane:", intersection_otherlane, "Intersection Offroad:", intersection_offroad, "*"*30)

                # working code
                current_location = self.ego.get_location()
                waypoint_info = self.map.get_waypoint(location=self.ego.get_location(), project_to_road=True)
                waypoint_info_lane = self.map.get_waypoint(location=self.ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Any)
                ego_location = self.ego.get_transform().location
                bounding_box = self.ego.bounding_box
                '''

                bounding_box_coordinates = self.ego.bounding_box.get_world_vertices(self.ego.get_transform())

                self.off_road_percentage, self.off_lane_percentage = self.compute_side_walk_opp_lane_infractions(bounding_box_coordinates)
                    
                time.sleep(3)
                self.collision_hist = []

                # Update timesteps
                self.time_step = 1
                self.reset_step += 1

                # Enable sync mode
                self.settings.synchronous_mode = True
                self.world.apply_settings(self.settings)

                # Set the initial speed to desired speed
                yaw = (self.ego.get_transform().rotation.yaw) * np.pi / 180.0
                init_speed = carla.Vector3D(
                    x=self.desired_speed * np.cos(yaw),
                    y=self.desired_speed * np.sin(yaw))
                self.ego.set_target_velocity(init_speed)
                self.world.tick()
                self.world.tick()

                # Reset action of last time step
                # TODO:[another kind of action]
                self.last_action = np.array([0.0, 0.0, 0.0])

                # End State variable initialized
                self.isCollided = False
                self.isTimeOut = False

                self.previous_distance_travelled = 0
                self.start_location = self.ego.get_transform().location

                self.previous_location = self.ego.get_transform().location
                self.distance_travelled = 0
                self.previous_velocity = self.ego.get_velocity()

                return self.current_image

            except Exception as e:
                self.logger.error("Env reset() error")
                self.logger.error(e)
                time.sleep(2)
                # Delete sensors, vehicles and walkers
                while self.actors:
                    (self.actors.pop()).destroy()
                self._make_carla_client('localhost', self.port)

    # Step Function - CARLA
    '''
    def step(self, action):
        try:
            # Assign acc/steer/brake to action signal
            # Ver. 1 input is the value of control signal
            # throttle_or_brake, steer = action[0], action[1]
            # if throttle_or_brake >= 0:
            #     throttle = throttle_or_brake
            #     brake = 0
            # else:
            #     throttle = 0
            #     brake = -throttle_or_brake

            # suspicious snippet of code
            current_action = np.array(action) + self.last_action
            current_action = np.clip(
                current_action, -1.0, 1.0)
            throttle_or_brake, steer = current_action

            if throttle_or_brake >= 0:
                throttle = throttle_or_brake
                brake = 0
            else:
                throttle = 0
                brake = -throttle_or_brake

            # Apply control
            act = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake))
            self.ego.apply_control(act)

            for _ in range(4):
                self.world.tick()

            # Update timesteps
            self.time_step += 1
            self.total_step += 1
            self.last_action = current_action

            # calculate reward
            isDone = self._terminal()
            current_reward = self.get_reward_bounding_boxes()

            return (current_reward, isDone), self.current_image

        except Exception as e:
            self.logger.error("Env step() error")
            self.logger.error(e)
            time.sleep(2)
            return (0.0, True), self.current_image
    '''

    # PPO Step function
    def step(self, action):
        try:
            current_action = np.array(action)
            throttle, brake, steer = current_action
            throttle = np.clip(throttle, 0, 1)
            brake = np.clip(brake, 0, 1)
            steer = np.clip(steer, -1, 1)

            # Apply control
            act = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake))
            self.ego.apply_control(act)

            for _ in range(4):
                self.world.tick()

            # Update timesteps
            self.time_step += 1
            self.total_step += 1
            self.last_action = current_action

            # calculate reward
            isDone = self._terminal()
            current_reward = self.get_reward_bounding_boxes()

            return self.current_image, current_reward, isDone, False, {}

        except Exception as e:
            self.logger.error("Env step() error")
            self.logger.error(e)
            time.sleep(2)
            return self.current_image, current_reward, isDone, False, {}

    def render(self, mode='human'):
        pass

    def close(self):
        while self.actors:
            (self.actors.pop()).destroy()

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        # ego_x, ego_y = self._get_ego_pos()

        # # If at destination
        # dest = self.dest
        # if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2) < 2.0:
        #     # print("Get destination! Episode Done.")
        #     self.logger.debug('Get destination! Episode cost %d steps in route %d.' % (self.time_step, self.route_id))
        #     # self.isSuccess = True
        #     return True

        # If collides
        if len(self.collision_hist) > 0:
            self.logger.debug(
                'Collision happened! Episode cost %d steps in route %d.' %
                (self.time_step, self.route_id))
            self.isCollided = True
            return True

        # If reach maximum timestep
        if self.time_step >= self.max_time_episode:
            self.logger.debug('Time out! Episode cost %d steps in route %d.' %
                              (self.time_step, self.route_id))
            self.isTimeOut = True
            return True

        return False

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker' or actor.type_id == 'sensor.camera.rgb' or actor.type_id == 'sensor.other.collision':
                        actor.stop()
                    actor.destroy()

    def _create_vehicle_bluepprint(self,
                                   actor_filter,
                                   color=None,
                                   number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
            actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
            bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [
                x for x in blueprints
                if int(x.get_attribute('number_of_wheels')) == nw
            ]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(
                    bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _set_carla_transform(self, pose):
        """Get a carla tranform object given pose.

        Args:
            pose: [x, y, z, pitch, roll, yaw].

        Returns:
            transform: the carla transform object
        """
        transform = carla.Transform()
        transform.location.x = pose[0]
        transform.location.y = pose[1]
        transform.location.z = pose[2]
        transform.rotation.pitch = pose[3]
        transform.rotation.roll = pose[4]
        transform.rotation.yaw = pose[5]
        return transform

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.

        Args:
            transform: the carla transform object.

        Returns:
            Bool indicating whether the spawn is successful.
        """
        vehicle = self.world.spawn_actor(self.ego_bp, transform)
        if vehicle is not None:
            self.actors.append(vehicle)
            self.ego = vehicle
            return True
        return False

    '''
    # TODO : Reset Traffic
    def _try_spawn_vehicle_at(self, transform):
        """Try to spawn traffic vehicles at specific transform

        Args:
            transform: the carla transform object
        Returns:
            Bool indicating whether spawn is successful
        """
        vehicle = self.world.spawn_actor(self.traffic_bp, transform)
        if vehicle is not None:
            self.actors.append(vehicle)
            return True
        return False
    '''

    def get_reward_bounding_boxes(self):
        '''
        # reward = 1000 * (d_cur - d_prev) + 0.05 * (v_cur - v_prev) - 0.00002 * (collision_damage_cur - collision_damage_prev) \
        # - 2 * (side_walk_intersection_cur - side_walk_intersection_prev) - 2 * (opposite_lane_intersection_cur - opposite_lane_intersection_prev)
        if self.isCollided:
            reward = -500
            return reward

        current_location = self.ego.get_transform().location
        distance_travelled = self.previous_location.distance(current_location)
        self.previous_location = current_location

        current_velocity = self.ego.get_velocity()
        prev_velocity_array = np.array([self.previous_velocity.x, self.previous_velocity.y])
        prev_velocity_norm = np.linalg.norm(prev_velocity_array)
        curr_velocity_array = np.array([current_velocity.x, current_velocity.y])
        curr_velocity_norm = np.linalg.norm(curr_velocity_array)
        velocity_diff = curr_velocity_norm - prev_velocity_norm
        self.previous_velocity = current_velocity

        bounding_box_coordinates = self.ego.bounding_box.get_world_vertices(self.ego.get_transform())
        curr_off_road_percentage, curr_off_lane_percentage = self.compute_side_walk_opp_lane_infractions(bounding_box_coordinates)
        
        side_walk_intersection_diff = curr_off_road_percentage - self.off_road_percentage
        off_lane_intersection_diff = curr_off_lane_percentage - self.off_lane_percentage

        self.off_road_percentage = curr_off_road_percentage
        self.off_lane_percentage = curr_off_lane_percentage

        # reward = 1000 * distance_travelled / 1000 + 0.05 * velocity_diff - 2 * side_walk_intersection_diff - 2 * off_lane_intersection_diff
        # reward = (1000 * distance_travelled / 1000)
        '''
        if self.isCollided:
            reward = -500
            return reward

        current_location = self.ego.get_transform().location
        distance_travelled_from_origin = abs(self.start_location.distance(current_location)) # meters
        current_velocity = self.ego.get_velocity() # m/s
        curr_velocity_array = np.array([current_velocity.x, current_velocity.y])
        curr_velocity_norm = np.linalg.norm(curr_velocity_array)
        r_step = 10
        reward = ((distance_travelled_from_origin - self.previous_distance_travelled) * 10) - abs(10 - curr_velocity_norm) + r_step
        self.previous_distance_travelled = distance_travelled_from_origin
        
        return reward

    def _make_carla_client(self, host, port):
        while True:
            try:
                self.logger.info("connecting to Carla server...")
                self.logger.info("Host: " + str(host) + " Port: " + str(port))
                self.client = carla.Client(host, port)
                self.client.set_timeout(10.0)

                # Set map
                if self.task_mode == 'Straight':
                    self.world = self.client.load_world('Town01')
                elif self.task_mode == 'Curve':
                    # self.world = self.client.load_world('Town01')
                    self.world = self.client.load_world('Town05')
                elif self.task_mode == 'Long':
                    self.world = self.client.load_world('Town01')
                    # self.world = self.client.load_world('Town02')
                elif self.task_mode == 'Lane':
                    # self.world = self.client.load_world('Town01')
                    self.world = self.client.load_world('Town05')
                elif self.task_mode == 'U_curve':
                    self.world = self.client.load_world('Town03')
                elif self.task_mode == 'Lane_test':
                    self.world = self.client.load_world('Town03')
                self.map = self.world.get_map()

                # Set weather
                self.world.set_weather(carla.WeatherParameters.ClearNoon)
                self.logger.info(
                    "Carla server port {} connected!".format(port))
                break
            except Exception as e:
                self.logger.error(e)
                self.logger.error(
                    'Fail to connect to carla-server...sleeping for 2')
                time.sleep(2)

    def _get_random_position_between(self, start, dest, transform):
        """
        get a random carla position on the line between start and dest
        """
        if self.task_mode == 'Straight':
            # s_x, s_y, s_z = start[0], start[1], start[2]
            # d_x, d_y, d_z = dest[0], dest[1], dest[2]

            # ratio = np.random.rand()
            # new_x = (d_x - s_x) * ratio + s_x
            # new_y = (d_y - s_y) * ratio + s_y
            # new_z = (d_z - s_z) * ratio + s_z

            # transform.location = carla.Location(x=new_x, y=new_y, z=new_z)
            start_location = carla.Location(x=start[0], y=start[1], z=0.22)
            ratio = float(np.random.rand() * 30)

            transform = self.map.get_waypoint(
                location=start_location).next(ratio)[0].transform
            transform.location.z = start[2]

        elif self.task_mode == 'Curve':
            start_location = carla.Location(x=start[0], y=start[1], z=0.22)
            ratio = float(np.random.rand() * 45)

            transform = self.map.get_waypoint(
                location=start_location).next(ratio)[0].transform
            transform.location.z = start[2]

        elif self.task_mode == 'Long' or self.task_mode == 'Lane':
            start_location = carla.Location(x=start[0], y=start[1], z=0.22)
            ratio = float(np.random.rand() * 60)

            transform = self.map.get_waypoint(
                location=start_location).next(ratio)[0].transform
            transform.location.z = start[2]

        return transform
