import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
from math import ceil
import cv2
from controller import Supervisor
from controllers.track_recognition import extract_track, process_track_into_line, get_track_properties, simulate_camera_view
from controllers.image_processing import bytes_to_image, save_image
from simpful import *

from skopt import gp_minimize

TRACK_DIRECTORY = "../../textures/RightAngle.png"

P_DIRECTION_STEER = 0
I_DIRECTION_STEER = 1
D_DIRECTION_STEER = 2

P_OFFSET_STEER = 3
I_OFFSET_STEER = 4
D_OFFSET_STEER = 5

P_DIRECTION_SPEED = 6
I_DIRECTION_SPEED = 7
D_DIRECTION_SPEED = 8

P_OFFSET_SPEED = 9
I_OFFSET_SPEED = 10
D_OFFSET_SPEED = 11

BASE_SPEED = 12
MAX_SPEED = 13

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
        self.wheels[0].setVelocity(0.2)
        self.wheels[1].setVelocity(0.2)
        self.wheels[2].setPosition(float('inf'))
        self.wheels[3].setPosition(float('inf'))
        self.set_steering_and_speed(0, 0)

    def set_steering_and_speed(self, speed_adjustment, steering_adjustment):
        # print(f"Speed: {speed_adjustment}, Steering: {steering_adjustment:.2f}")
        speed = 0
        steering = 0

        speed += speed_adjustment
        speed = max(speed, 1)
        speed = min(speed, 10)
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

        # print("[Notification]: Robot position and rotation reset to initial values.")

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


class LineFollowingEnv():
    def __init__(self, robot_controller, use_camera=True):
        super(LineFollowingEnv, self).__init__()
        self.robot_controller = robot_controller
        self.use_camera = use_camera

        self.track_image = cv2.imread(TRACK_DIRECTORY, cv2.IMREAD_GRAYSCALE)
        self.track_image = np.array(self.track_image, dtype=np.uint8)

        self.step_count = 0  # Track steps to manage observation logging interval
        self.observation_log_interval = 10  # Print observations every 10 steps

    def reset(self):
        # print("[Notification]: Environment reset.")
        self.step_count = 0
        self.robot_controller.reset_robots_position_and_rotation()
        self.robot_controller.set_steering_and_speed(0, 0)
        # self.robot_controller.run_simulation_for_time(0.5)
        return self._get_observation()

    def step(self, speed_adjustment, steering_adjustment):
        
        # Apply the calculated speed and steering to the robot
        self.robot_controller.set_steering_and_speed(speed_adjustment, steering_adjustment)
        self.robot_controller.run_one_step()

        # Get the observation after applying the action
        observation = self._get_observation()
        reward, done = self._calculate_reward(observation)
        # if done:
            # print("[Notification]: Episode ended. Robot is out of track.")
        return observation, reward, done, {}
    
    def _get_camera_view(self):
        image_bytes = self.robot_controller.camera.getImage()
        width = self.robot_controller.camera.getWidth()
        height = self.robot_controller.camera.getHeight()
        image = bytes_to_image(image_bytes, width, height)
        return image
    
    def _simulate_camera_view(self):
        position, rotation = self.robot_controller.get_robots_position_and_rotation(self.track_image)
        # print(f"Position: {position}, Rotation: {rotation}")
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
        track_direction, track_offset = get_track_properties(m, c, extracted_track.shape)

        self.step_count += 1
        # if self.step_count % self.observation_log_interval == 0:
            # print(f"[Observation]: Direction (m): {m:.2f}, Offset from middle (c): {track_offset:.2f}, Speed: {speed:.2f}")

        return np.array([track_direction, track_offset, speed], dtype=np.float32)

    def _calculate_reward(self, observation):
        track_direction, track_offset, speed = observation
        max_offset = 1

        if track_offset == 0 and track_direction == 0:
            reward = -10
            # print(f"[Notification]: Robot is off-track. Offset: {track_offset:.2f}")
            # print(f"[Observation]: Reward: {reward:.2f}")
            return reward, True  # Large penalty for going off track and terminate episode

        reward = 0.1  # Base reward
        reward += (max_offset - abs(track_offset)) * 10 # deviation from line reward
        reward += (speed - 1) * 0.1 # Speed reward
        
        # if self.step_count % self.observation_log_interval == 0:
            # print(f"[Observation]: Reward: {reward:.2f}")

        return reward, False

class FuzzyController:
    def __init__(self, env, genotype_bounds, evaluation_steps=1000):
        self.env = env
        self.genotype_bounds = [(0, 0) for _ in range(len(genotype_bounds))]
        for i, bound in genotype_bounds.items():
            self.genotype_bounds[i] = bound
        self.prev_offset = 0
        self.prev_direction = 0
        self.evaluation_steps = evaluation_steps
        self.episode = 0

    def calculate_speed(self, genotype, observation):
        track_direction, track_offset, speed = observation
        offset = 1 - abs(track_offset)
        direction = 1 - abs(track_direction)

        offset_integral = offset + self.prev_offset
        offset_derivative = offset - self.prev_offset

        direction_integral = direction + self.prev_direction
        direction_derivative = direction - self.prev_direction

        base_speed = genotype[BASE_SPEED]
        
        max_speed_gain = genotype[P_DIRECTION_SPEED] + genotype[I_DIRECTION_SPEED] + genotype[D_DIRECTION_SPEED] + \
                                genotype[P_OFFSET_SPEED] + genotype[I_OFFSET_SPEED] + genotype[D_OFFSET_SPEED]

        if max_speed_gain == 0:
            return base_speed

        min_max_speed_diff = genotype[MAX_SPEED] - base_speed

        speed = genotype[P_DIRECTION_SPEED] * direction + \
                genotype[I_DIRECTION_SPEED] * direction_integral + \
                genotype[D_DIRECTION_SPEED] * direction_derivative + \
                genotype[P_OFFSET_SPEED] * offset + \
                genotype[I_OFFSET_SPEED] * offset_integral + \
                genotype[D_OFFSET_SPEED] * offset_derivative

        speed = base_speed + (speed / max_speed_gain) * min_max_speed_diff

        # print(f"Speed: {speed:.2f}, Base Speed: {base_speed:.2f}, Max Speed Gain: {max_speed_gain:.2f}, Min Max Speed Diff: {min_max_speed_diff:.2f}")

        return speed

    def calculate_steer(self, genotype, observation):
        direction, offset, speed = observation

        offset_integral = offset + self.prev_offset
        offset_derivative = offset - self.prev_offset

        direction_integral = direction + self.prev_direction
        direction_derivative = direction - self.prev_direction

        steer = genotype[P_DIRECTION_STEER] * direction + \
                genotype[I_DIRECTION_STEER] * direction_integral + \
                genotype[D_DIRECTION_STEER] * direction_derivative + \
                genotype[P_OFFSET_STEER] * offset + \
                genotype[I_OFFSET_STEER] * offset_integral + \
                genotype[D_OFFSET_STEER] * offset_derivative

        steer *= -1

        return steer

    def evaluate_genotype(self, genotype):
        observation = self.env.reset()
        done = False
        total_reward = 0

        print(f"Episode: {self.episode}")
        self.episode += 1

        for _ in range(self.evaluation_steps):
            # print(f"Offset: {observation[1]:.2f}, Direction: {observation[0]:.2f}")
            speed_adjustment = self.calculate_speed(genotype, observation)
            steering_adjustment = self.calculate_steer(genotype, observation)

            self.prev_offset = observation[1]
            self.prev_direction = observation[0]


            observation, reward, done, _ = self.env.step(speed_adjustment, steering_adjustment)
            total_reward += reward

            if done:
                break

        return -1 * total_reward

    def train(self, episodes):
        result = gp_minimize(
            self.evaluate_genotype,
            self.genotype_bounds,
            n_calls=episodes,
            random_state=0,
            n_jobs=1
        )

        print(f"Best Genotype: {result.x}")
        print(f"Best Reward: {result.fun}")
        return result.x


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


    # FS = FuzzySystem()

    # Tra_1 = TrapezoidFuzzySet(0, 0, 20, 30, "miękka")
    # Tra_2 = TrapezoidFuzzySet(20, 30, 50, 60, "srednioTwarda")
    # Tra_3 = TrapezoidFuzzySet(50, 60, 80, 90, "twarda")
    # Tra_4 = TrapezoidFuzzySet(80, 90, 100, 100, "bardzoTwarda")
    # FS.add_linguistic_variable("wat_qua", LinguisticVariable([Tra_1, Tra_2, Tra_3, Tra_4], universe_of_discourse=[0, 100]))
    # FS.plot_variable("wat_qua")

    # Tra_1 = TrapezoidFuzzySet(0, 0, 5, 10, "bardzoMała")
    # Tra_2 = TrapezoidFuzzySet(5, 10, 20, 25, "mała")
    # Tra_3 = TriangleFuzzySet(20, 25, 30, "srednia")
    # Tra_4 = TrapezoidFuzzySet(25, 30, 40, 45, "duża")
    # Tra_5 = TrapezoidFuzzySet(40, 45, 50, 50, "bardzoDuża")
    # FS.add_linguistic_variable("coff_grind",  LinguisticVariable([Tra_1, Tra_2, Tra_3, Tra_4, Tra_5], universe_of_discourse=[0, 50]))
    # FS.plot_variable("coff_grind")

    # Tra_1 = TrapezoidFuzzySet(0, 0, 150, 200, "mała")
    # Tra_2 = TrapezoidFuzzySet(150, 200, 250, 300, "srednia")
    # Tra_3 = TrapezoidFuzzySet(250, 300, 350, 400, "duża")
    # Tra_4 = TrapezoidFuzzySet(350, 400, 500, 500, "bardzoDuża")
    # FS.add_linguistic_variable("coff_amout", LinguisticVariable([Tra_1, Tra_2, Tra_3, Tra_4], universe_of_discourse=[0,500]))
    # FS.plot_variable("coff_amout", outputfile="coff_amout.png")

    # Define genotype bounds
    genotype_bounds = {
        P_DIRECTION_STEER: (0, 1),
        I_DIRECTION_STEER: (0, 1),
        D_DIRECTION_STEER: (0, 1),

        P_OFFSET_STEER: (0, 1),
        I_OFFSET_STEER: (0, 1),
        D_OFFSET_STEER: (0, 1),

        P_DIRECTION_SPEED: (0, 1),
        I_DIRECTION_SPEED: (0, 1),
        D_DIRECTION_SPEED: (0, 1),

        P_OFFSET_SPEED: (0, 1),
        I_OFFSET_SPEED: (0, 1),
        D_OFFSET_SPEED: (0, 1),

        BASE_SPEED: (0, 3),
        MAX_SPEED: (3, 10),
    }

    # Initialize PID controller and train
    controller = FuzzyController(env, genotype_bounds)
    controller.train(episodes=2000)

