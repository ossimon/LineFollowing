import numpy as np
from controller import Supervisor
from PIL import Image

# camera width is 1280 pixels
TIME_STEP = 64
SAMPLING_PERIOD = 100

def save_image(image, filename):
    # convert image to numpy array
    image_array = np.frombuffer(image, dtype=np.uint8)
    # reshape array to match the shape of the image
    image_array = image_array.reshape((camera.getHeight(), camera.getWidth(), 4))
    # swap red and blue channels
    image_array = image_array[:, :, [2, 1, 0, 3]]
    # save image
    image = Image.fromarray(image_array, mode='RGBA')
    image.save(filename)

def calculate_line_expected_position(width, bottom_row_of_pixels):
    bottom_row_of_pixels = np.array(bottom_row_of_pixels, dtype=np.float64)
    darkest_pixel = bottom_row_of_pixels.min()
    brightest_pixel = bottom_row_of_pixels.max()
    
    # Handle the case where all pixels are identical
    if darkest_pixel == brightest_pixel:
        darkness_intensities = np.ones(width) / width
    else:
        brightness_intensity = (bottom_row_of_pixels - darkest_pixel) / (brightest_pixel - darkest_pixel)
        darkness_intensities = 1 - brightness_intensity
        darkness_intensities /= darkness_intensities.sum()

    line_expected_position = np.dot(darkness_intensities, np.arange(width))
    return line_expected_position

def decide_action(bottom_row_of_pixels, width, speed_multiplier, steering_multiplier):
    
    bottom_row_of_pixels = np.array(bottom_row_of_pixels, dtype=np.float64)
    bottom_row_of_pixels[bottom_row_of_pixels < 50] = 0
    bottom_row_of_pixels[bottom_row_of_pixels >= 50] = 255

    line_expected_position = calculate_line_expected_position(width, bottom_row_of_pixels)

    proportion = -1 * (line_expected_position / width - 0.5)
    speed = 2
    steering = 0

    speed += speed_multiplier * (0.5 - abs(proportion))
    steering += steering_multiplier * proportion

    return speed, steering

def run_robot(translation_field, speed_multiplier, steering_multiplier):
    distance = 0

    while robot.step(TIME_STEP) != -1:
        robot_z = translation_field.getSFVec3f()[2]
        if robot_z < 0 or robot_z > 0.5:
            break

        image = camera.getImage()
        width = camera.getWidth()
        row = camera.getHeight() - 1

        bottom_row_of_pixels = [
            camera.imageGetGray(image, width, column, row) 
            for column in range(width)
        ]

        speed, steering = decide_action(bottom_row_of_pixels, width, speed_multiplier, steering_multiplier)

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

    speed_multiplier = 5
    steering_multiplier = 1

    for i in range(1, 10):
        for j in range(1, 10):
            speed_multiplier = i
            steering_multiplier = j

            reset_robot(translation_field, rotation_field, initial_position, initial_rotation)
            fitness = run_robot(translation_field, speed_multiplier, steering_multiplier)
            print(f"Fitness: {fitness}")
