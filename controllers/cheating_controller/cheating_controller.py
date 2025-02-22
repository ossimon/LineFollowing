import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from controller import Supervisor
import cv2
import numpy as np
from controllers.track_recognition import process_track_into_line, get_track_properties

TRACK_DIRECTORY = "../../textures/hard_track.png"
TRACK_BOUNDARIES_DIRECTORY = "../../textures/hard_track_with_grass.png"

TIME_STEP = 128
SAMPLING_PERIOD = 100

wheels = [None] * 4
last_steering = 0
last_speed = 0

track = None
boundaries = None

MAKE_SS = True
SS_INTERVAL = 18 # in seconds
SS_ONLY_ONCE = True

def run_simulation_for_time(time_to_run):
    step_count = 0
    steps_to_run = time_to_run * 1000 / TIME_STEP

    while robot.step(TIME_STEP) != -1:
        if step_count >= steps_to_run:
            break
        step_count += 1

def set_steering_and_speed(steering, speed):
    global last_steering, last_speed
    last_speed = speed
    last_steering = steering
    wheels[0].setPosition(steering)
    wheels[1].setPosition(steering)
    wheels[2].setVelocity(speed)
    wheels[3].setVelocity(speed)

def decide_action(translation_field, rotation_field, speed_multiplier, steering_multiplier):
    global track 
    robot_x = translation_field.getSFVec3f()[0]
    robot_y = -1 * translation_field.getSFVec3f()[1]
    robot_x += 2.5
    robot_y += 2.5
    robot_x /= 5
    robot_y /= 5
    robot_x *= track.shape[1] # 0 <= robot_x < track.shape[1]
    robot_y *= track.shape[0] # 0 <= robot_y < track.shape[0]
    robot_x = int(robot_x)
    robot_y = int(robot_y)

    rotation_list = rotation_field.getSFRotation()
    rotation = rotation_list[3]
    rotation *= -1 if rotation_list[2] < 0 else 1
    rotation += np.pi
    print(f"Robot position: {robot_x}, {robot_y}, rotation: {rotation}")

    extracted_track = track[robot_y-10:robot_y+10, robot_x-10:robot_x+10]

    m, c = process_track_into_line(extracted_track)

    track_direction, track_offset_from_the_middle = get_track_properties(m, c, extracted_track.shape)

    # print(f"m: {m:.2f}, c: {c:.2f}")
    # print(f"Track direction: {track_direction:.2f}, track offset from the middle: {track_offset_from_the_middle:.2f}")

    proportion = track_offset_from_the_middle
    speed = 2
    steering = 0

    speed += speed_multiplier * (0.5 - abs(proportion))
    steering -= steering_multiplier * proportion

    return speed, steering

def out_ouf_bounds(translation_field):
    robot_z = translation_field.getSFVec3f()[2]
    robot_x = translation_field.getSFVec3f()[0]
    robot_y = -1 * translation_field.getSFVec3f()[1]
    robot_x += 2.5
    robot_y += 2.5
    robot_x /= 5
    robot_y /= 5
    robot_x *= boundaries.shape[1]
    robot_y *= boundaries.shape[0]
    if robot_z < 0 or robot_z > 15 or robot_x < 0 or robot_x > boundaries.shape[1] or robot_y < 0 or robot_y > boundaries.shape[0]:
        return True, "escaped the map"
    if boundaries[int(robot_y), int(robot_x)] == 1:
        return True, "too far from the track"
    return False, ""

def run_robot(translation_field, rotation_field, speed_multiplier, steering_multiplier, timeout=60):
    step_count = 0
    distance = 0
    steps_to_run = timeout * 1000 / TIME_STEP

    global MAKE_SS, SS_ONLY_ONCE

    while robot.step(TIME_STEP) != -1:
        if step_count >= steps_to_run:
            # print("Timeout")
            break

        out_of_bounds, reason = out_ouf_bounds(translation_field)
        if out_of_bounds:
            # print(f"Out of bounds: {reason}")
            break

        speed, steering = decide_action(translation_field, rotation_field, speed_multiplier, steering_multiplier)

        set_steering_and_speed(steering, speed)

        distance += speed * TIME_STEP / 1000.0
        step_count += 1

    return distance

def prepare_wheels():
    global wheels
    wheel_names = ["left_steer", "right_steer", "wheel3", "wheel4"]
    for i, name in enumerate(wheel_names):
        wheels[i] = robot.getDevice(name)

    # set the speed at which the front wheels turn left and right
    wheels[0].setVelocity(.5)
    wheels[1].setVelocity(.5)
    # set the destination of the rear wheels to infinity
    wheels[2].setPosition(float('inf'))
    wheels[3].setPosition(float('inf'))

    set_steering_and_speed(0, 0)
    run_simulation_for_time(0.1)

def reset_robots_position_and_rotation(translation_field, rotation_field, initial_position, initial_rotation):
    translation_field.setSFVec3f(initial_position)
    rotation_field.setSFRotation(initial_rotation)

    robot_node = robot.getFromDef("ROBOT")
    robot_node.resetPhysics()

if __name__ == "__main__":

    robot = Supervisor()
    speed_multiplier = 5
    steering_multiplier = 2
    
    robot_node = robot.getFromDef("ROBOT")
    translation_field = robot_node.getField("translation")
    rotation_field = robot_node.getField("rotation")
    
    # read the track image
    track_image = cv2.imread(TRACK_DIRECTORY, cv2.IMREAD_GRAYSCALE)
    track = np.array(track_image, dtype=np.uint8)
    # print(f"Min: {np.min(track)}, Max: {np.max(track)}")
    # print(f"Shape: {track.shape}")
    # read the track boundaries image
    track_boundaries_image = cv2.imread(TRACK_BOUNDARIES_DIRECTORY)
    boundaries = np.zeros((track_boundaries_image.shape[0], track_boundaries_image.shape[1]), dtype=np.uint8)
    boundaries[track_boundaries_image[:, :, 0] == 255] = 1

    itinial_position = translation_field.getSFVec3f()
    initial_rotation = rotation_field.getSFRotation()

    for try_number in range(5):
        # print(f"Try number {try_number}")

        prepare_wheels()
        reset_robots_position_and_rotation(translation_field, rotation_field, itinial_position, initial_rotation)

        fitness = run_robot(translation_field, rotation_field, speed_multiplier, steering_multiplier, timeout=120)
        # print(f"Fitness: {fitness}")

