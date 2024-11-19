import numpy as np
from controller import Supervisor, Motor
import math
from time import time

TIME_STEP = 64
SAMPLING_PERIOD = 100

def calculate_expected_position(width, row_image):
    row_image = np.array(row_image, dtype=np.float64)
    darkest_pixel = row_image.min()
    brightest_pixel = row_image.max()
    
    # Handle the case where all pixels are identical
    if darkest_pixel == brightest_pixel:
        darkness_intensities = np.ones(width) / width
    else:
        brightness_intensity = (row_image - darkest_pixel) / (brightest_pixel - darkest_pixel)
        darkness_intensities = 1 - brightness_intensity
        darkness_intensities /= darkness_intensities.sum()

    expected_position = np.dot(darkness_intensities, np.arange(width))
    return expected_position

def run_robot(translation_field, speed_multipliers, steering_multipliers):
    distance = 0
    last_expected_position = 640
    last_proportion = 0

    while robot.step(TIME_STEP) != -1:
        
        robot_z = translation_field.getSFVec3f()[2]
        if robot_z < 0 or robot_z > 0.5:
            break

        image = camera.getImage()
        width = camera.getWidth()
        row = camera.getHeight() - 1

        row_image = [
            camera.imageGetGray(image, width, column, row) 
            for column in range(width)
        ]

        expected_position = calculate_expected_position(width, row_image)

        if math.isnan(expected_position):
            if math.isnan(last_expected_position):
                break
            last_expected_position = expected_position
            continue

        proportion = -1 * (expected_position / width - 0.5)
        derivative = proportion + last_proportion

        speed = 2
        steering = 0

        if not math.isnan(expected_position):
            speed += speed_multipliers[0] * (0.5 - abs(proportion))
            steering += steering_multipliers[0] * proportion + steering_multipliers[1] * derivative

        wheels[0].setPosition(steering)
        wheels[1].setPosition(steering)
        wheels[2].setVelocity(speed)
        wheels[3].setVelocity(speed)

        distance += speed * TIME_STEP / 1000.0

    return distance

def reset_robot(translation_field, rotation_field, initial_position, initial_rotation):
    wheel_names = ["left_steer", "right_steer", "wheel3", "wheel4"]
    for i, name in enumerate(wheel_names):
        wheels[i] = robot.getDevice(name)
        if i < 2:
            wheels[i].setPosition(0)
            wheels[i].setVelocity(3)
        else:
            wheels[i].setPosition(float('inf'))
            wheels[i].setVelocity(0)

    translation_field.setSFVec3f(initial_position)
    rotation_field.setSFRotation(initial_rotation)

if __name__ == "__main__":
    robot = Supervisor()
    robot_node = robot.getFromDef("ROBOT")

    translation_field = robot_node.getField("translation")
    rotation_field = robot_node.getField("rotation")

    initial_position = list(translation_field.getSFVec3f())
    initial_rotation = list(rotation_field.getSFRotation())

    camera = robot.getDevice("camera")
    camera.enable(SAMPLING_PERIOD)

    wheels = [None] * 4

    speed_multipliers = [5, 0]
    steering_multipliers = [1, 0]

    for i in range(1, 10):
        for j in range(1, 10):
            speed_multipliers[0] = i
            steering_multipliers[0] = j

            reset_robot(translation_field, rotation_field, initial_position, initial_rotation)
            fitness = run_robot(translation_field, speed_multipliers, steering_multipliers)
            print(f"Fitness: {fitness}")
