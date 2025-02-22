import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
from math import ceil
import cv2
from controller import Supervisor
from controllers.track_recognition import extract_track, process_track_into_line, get_track_properties, simulate_camera_view
from controllers.image_processing import bytes_to_image, save_image
from controllers.q_learning import  QLearner

TRACK_DIRECTORY = "../../textures/EasyCurve.png"

class RobotController:
    def __init__(self, time_step=64, use_camera=True):
        self.robot = Supervisor()
        self.time_step = time_step
        if use_camera:
            self.camera = self.robot.getDevice("camera")
            self.camera.enable(time_step)

        floor_node = self.robot.getFromDef("FLOOR")
        self.floor_size = floor_node.getField("size").getSFVec3f()

        self.translation_field = self.robot.getFromDef("ROBOT").getField("translation")
        self.rotation_field = self.robot.getFromDef("ROBOT").getField("rotation")

        # Store the initial position and rotation of the robot
        self.initial_position = self.translation_field.getSFVec3f()
        self.initial_rotation = self.rotation_field.getSFRotation()
        
        self.wheels = [self.robot.getDevice(name) for name in ["left_steer", "right_steer", "wheel3", "wheel4"]]
        self.prepare_wheels()

    def prepare_wheels(self):
        self.wheels[0].setVelocity(0.5)
        self.wheels[1].setVelocity(0.5)
        self.wheels[2].setPosition(float('inf'))
        self.wheels[3].setPosition(float('inf'))
        self.set_steering_and_speed(0, 0)

    def set_steering_and_speed(self, speed_adjustment, steering_adjustment):
        speed = 2
        steering = 0

        speed += speed_adjustment
        steering += steering_adjustment * 0.5

        self.wheels[0].setPosition(steering)
        self.wheels[1].setPosition(steering)
        self.wheels[2].setVelocity(speed)
        self.wheels[3].setVelocity(speed)

    def reset_robots_position_and_rotation(self):
        self.translation_field.setSFVec3f(self.initial_position)
        self.rotation_field.setSFRotation(self.initial_rotation)

        robot_node = self.robot.getFromDef("ROBOT")
        robot_node.resetPhysics()
        print("[Notification]: Robot position and rotation reset to initial values.")

    def get_robots_position_and_rotation(self, track):
        robot_x = self.translation_field.getSFVec3f()[0]
        robot_y = -1 * self.translation_field.getSFVec3f()[1]
        robot_x = robot_x / self.floor_size[0] + 0.5
        robot_y = robot_y / self.floor_size[1] + 0.5
        robot_x *= track.shape[1] # 0 <= robot_x < track.shape[1]
        robot_y *= track.shape[0] # 0 <= robot_y < track.shape[0]
        robot_x = int(robot_x)
        robot_y = int(robot_y)

        rotation_list = self.rotation_field.getSFRotation()
        rotation = rotation_list[3]
        rotation *= -1 if rotation_list[2] < 0 else 1
        rotation += np.pi
        
        return (robot_x, robot_y), rotation

    def run_simulation_for_time(self, duration):
        steps = ceil(duration * 1000 / self.time_step)
        for _ in range(steps):
            self.robot.step(self.time_step)

    def run_one_step(self):
        self.robot.step(self.time_step)

# Line Following Environment
import gym
from gym import spaces

class LineFollowingEnv(gym.Env):
    def __init__(self, robot_controller, use_camera=True):
        super(LineFollowingEnv, self).__init__()
        self.robot_controller = robot_controller
        self.use_camera = use_camera

        self.track_image = cv2.imread(TRACK_DIRECTORY, cv2.IMREAD_GRAYSCALE)
        self.track_image = np.array(self.track_image, dtype=np.uint8)

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0]),  # Speed lower bound is 0
            high=np.array([np.inf, np.inf, np.inf]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)  # Discretized actions
        self._map_actions()

        self.step_count = 0  # Track steps to manage observation logging interval
        self.observation_log_interval = 10  # Print observations every 10 steps

    def _map_actions(self):
        self.actions = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (0, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1),
        ]

    def reset(self):
        print("[Notification]: Environment reset.")
        self.step_count = 0
        self.robot_controller.reset_robots_position_and_rotation()
        return self._get_observation()

    def step(self, action):
        speed_adjustment, steering_adjustment = self.actions[action]
        
        # Apply the calculated speed and steering to the robot
        self.robot_controller.set_steering_and_speed(speed_adjustment, steering_adjustment)
        self.robot_controller.run_one_step()

        # Get the observation after applying the action
        observation = self._get_observation()
        reward, done = self._calculate_reward(observation)
        if done:
            print("[Notification]: Episode ended. Robot is out of track.")
        return observation, reward, done, {}
    
    def _get_camera_view(self):
        image_bytes = self.robot_controller.camera.getImage()
        width = self.robot_controller.camera.getWidth()
        height = self.robot_controller.camera.getHeight()
        image = bytes_to_image(image_bytes, width, height)
        return image
    
    def _simulate_camera_view(self):
        position, rotation = self.robot_controller.get_robots_position_and_rotation(self.track_image)
        return simulate_camera_view(
            self.track_image,
            position,
            np.degrees(rotation) + 90,
            self.track_image.shape[0] // 15
        )

    def _get_observation(self):
        wheel3_speed = self.robot_controller.wheels[2].getVelocity()
        wheel4_speed = self.robot_controller.wheels[3].getVelocity()
        speed = max(0, (wheel3_speed + wheel4_speed) / 2)

        if self.use_camera:
            camera_view = self._get_camera_view()
            extracted_track = extract_track(camera_view)
        else:
            camera_view = self._simulate_camera_view()
            extracted_track = np.where(camera_view > 0, 0, 1).astype(np.uint8)

        m, c = process_track_into_line(extracted_track)
        track_direction, track_offset_from_middle = get_track_properties(m, c, extracted_track.shape)

        self.step_count += 1
        if self.step_count % self.observation_log_interval == 0:
            print(f"[Observation]: Slope (m): {m:.2f}, Offset from middle (c): {track_offset_from_middle:.2f}, Speed: {speed:.2f}")

        return np.array([track_direction, track_offset_from_middle, speed], dtype=np.float32)

    def _calculate_reward(self, observation):
        track_direction, track_offset_from_middle, speed = observation
        max_offset = 1

        if track_offset_from_middle == 0 and track_direction == 0:
            reward = -10
            print(f"[Notification]: Robot is off-track. Offset: {track_offset_from_middle:.2f}")
            print(f"[Observation]: Reward: {reward:.2f}")
            return reward, True  # Large penalty for going off track and terminate episode

        reward = 0.1  # Base reward
        reward += (max_offset - abs(track_offset_from_middle)) * 10 # deviation from line reward
        reward += (speed - 1) * 0.1 # Speed reward
        
        if self.step_count % self.observation_log_interval == 0:
            print(f"[Observation]: Reward: {reward:.2f}")

        return reward, False

if __name__ == "__main__":
    # Choose between using the camera or
    # simulating the its view (faster, but less realistic)
    use_camera = False

    # Initialize the robot controller
    robot_controller = RobotController(
        time_step=128,
        use_camera=use_camera
    )

    # Create the environment
    env = LineFollowingEnv(robot_controller, use_camera)

    # Define Q-learning configuration
    config = {
        "buckets_sizes": [[0.35, 0.125, 0.05, 0.125, 0.35], [0.3, 0.15, 0.1, 0.15, 0.3], [0.33, 0.34, 0.33]],
        "lower_bounds": [-1, -0.5, 0],
        "upper_bounds": [1, 0.5, 5],
        "epsilon_max": 0.8,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.15,
        "alpha_max": 0.5,
        "alpha_min": 0.1,
        "alpha_decay": 0.15,
        "gamma": 0.99,
    }

    # Initialize Q-learner and train
    learner = QLearner(env, config)
    learner.train(episodes=100, max_step=1000)

    # Test the trained model
    print("Testing trained model...")
    learner.test(episodes=2)