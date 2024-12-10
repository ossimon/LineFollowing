import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from controller import Supervisor
import cv2
import numpy as np
from controllers.track_recognition import extract_track, process_track_into_line, get_track_properties
from controllers.image_processing import bytes_to_image, save_image

TRACK_BOUNDARIES_DIRECTORY = "../../textures/hard_track_with_grass.png"

TIME_STEP = 64
SAMPLING_PERIOD = 100

wheels = [None] * 4
last_steering = 0
last_speed = 0

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

def decide_action(image, speed_multiplier, steering_multiplier):
    extracted_track = extract_track(image)
    
    m, c = process_track_into_line(extracted_track)

    if m == 0 and c == 0:
        save_image(image, "c0m0img.png")
        save_image(extracted_track, "c0m0track.png")
        for pixel in image[-1]:
            print(pixel)
        print(extracted_track[-1])

    track_direction, track_offset_from_the_middle = get_track_properties(m, c, extracted_track.shape)

    print(f"m: {m:.2f}, c: {c:.2f}")
    print(f"Track direction: {track_direction:.2f}, track offset from the middle: {track_offset_from_the_middle:.2f}")

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

def run_robot(translation_field, speed_multiplier, steering_multiplier, timeout=60):
    step_count = 0
    distance = 0
    steps_to_run = timeout * 1000 / TIME_STEP
    steps_to_sceenshot = SS_INTERVAL * 1000 // TIME_STEP
    screenshot_count = 0

    global MAKE_SS, SS_ONLY_ONCE

    while robot.step(TIME_STEP) != -1:
        if step_count >= steps_to_run:
            print("Timeout")
            break

        out_of_bounds, reason = out_ouf_bounds(translation_field)
        if out_of_bounds:
            print(f"Out of bounds: {reason}")
            break

        image_bytes = camera.getImage()
        width = camera.getWidth()
        height = camera.getHeight()

        image = bytes_to_image(image_bytes, width, height)

        if (MAKE_SS and step_count % steps_to_sceenshot == 0):
            print("Taking screenshot")
            save_image(image, f"ss_{SS_INTERVAL}.png")
            screenshot_count += 1
            if SS_ONLY_ONCE:
                MAKE_SS = False

        speed, steering = decide_action(image, speed_multiplier, steering_multiplier)

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

    epsilon = 0.01
    while last_speed > epsilon or last_steering > epsilon:
        set_steering_and_speed(last_speed * 0.9, last_steering * 0.9)
        run_simulation_for_time(0.1)

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

    camera = robot.getDevice("camera")
    camera.enable(SAMPLING_PERIOD)
    
    # read the track grass image
    track_grass_image = cv2.imread(TRACK_BOUNDARIES_DIRECTORY)
    
    boundaries = np.zeros((track_grass_image.shape[0], track_grass_image.shape[1]), dtype=np.uint32)
    boundaries[track_grass_image[:, :, 0] == 255] = 1

    itinial_position = translation_field.getSFVec3f()
    initial_rotation = rotation_field.getSFRotation()

    for try_number in range(5):
        print(f"Try number {try_number}")

        prepare_wheels()
        reset_robots_position_and_rotation(translation_field, rotation_field, itinial_position, initial_rotation)

        fitness = run_robot(translation_field, speed_multiplier, steering_multiplier, timeout=120)
        print(f"Fitness: {fitness}")

