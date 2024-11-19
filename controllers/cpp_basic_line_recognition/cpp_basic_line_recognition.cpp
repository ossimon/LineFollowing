#include <webots/DistanceSensor.hpp>
#include <webots/Motor.hpp>
#include <webots/Robot.hpp>
#include <webots/Camera.hpp>
#include <webots/Supervisor.hpp>
#include <iostream>
#include <cmath>
#include <array>

#define TIME_STEP 64
#define SAMPLING_PERIOD 100

using namespace webots;
using namespace std;

Supervisor *robot;
Camera *camera;
Motor *wheels[4];

double calculate_expected_position(int width, const vector<unsigned char> &row_image) {
  int darkest_pixel = 255;
  for (int column = 0; column < width; column++) {
    if (row_image[column] < darkest_pixel) {
      darkest_pixel = row_image[column];
    }
  }
  int brightest_pixel = 0;
  for (int column = 0; column < width; column++) {
    if (row_image[column] > brightest_pixel) {
      brightest_pixel = row_image[column];
    }
  }

  double darkness_intensities[width];
  for (int column = 0; column < width; column++) {
    double brightness_intensity = static_cast<double>(row_image[column] - darkest_pixel) / static_cast<double>(brightest_pixel - darkest_pixel);
    darkness_intensities[column] = 1 - brightness_intensity;
  }

  double sum = 0;
  for (int column = 0; column < width; column++) {
    sum += darkness_intensities[column];
  }
  for (int column = 0; column < width; column++) {
    darkness_intensities[column] /= sum;
  }

  double expected_position = 0;
  for (int column = 0; column < width; column++) {
    expected_position += darkness_intensities[column] * column;
  }

  return expected_position;
} 

double run_robot(Field *translation_field, const array<double, 2> &speed_multipliers, const array<double, 2> &steering_multipliers) {
  double distance = 0;
  double last_expected_position = 640;
  double last_proportion = 0;

  while (robot->step(TIME_STEP) != -1) {

    // if we fell off the platform, return time
    double robot_z = translation_field->getSFVec3f()[2];
    // cout << "Robot Z: " << robot_z << endl;
    if (robot_z < 0 || robot_z > 0.5) {
      break;
    }

    const unsigned char *image = camera->getImage();
    int width = camera->getWidth();
    int row = camera->getHeight() - 1;
    vector<unsigned char> row_image(width);

    for (int column = 0; column < width; column++) {
      row_image[column] = camera->imageGetGray(image, width, column, row);
    }

    double expected_position = calculate_expected_position(width, row_image);

    // if both expected position and last expected position are NaN, return time
    if (expected_position != expected_position) {
      if (last_expected_position != last_expected_position) {
        break;
      }
      last_expected_position = expected_position;
      continue;
    }

    double proportion = -1 * (expected_position / width - 0.5);
    double derivative = proportion + last_proportion;

    double speed = 2;
    double steering = 0;

    if (expected_position == expected_position) {
      speed += speed_multipliers.at(0) * (0.5 - abs(proportion));
      steering += steering_multipliers.at(0) * proportion + steering_multipliers.at(1) * derivative;
    }
    
    wheels[0]->setPosition(steering);
    wheels[1]->setPosition(steering);
    wheels[2]->setVelocity(speed);
    wheels[3]->setVelocity(speed);

    distance += speed * (double) TIME_STEP / 1000.0;
  }

  return distance;
}

void reset_robot(Field *translation_field, Field *rotation_field, const array<double, 3> &initial_position, const array<double, 4> &initial_rotation) {
  // reset the wheels
  string wheels_names[4] = {"left_steer", "right_steer", "wheel3", "wheel4"};
  for (int i = 0; i < 4; i++) {
    wheels[i] = robot->getMotor(wheels_names[i]);
    double position = INFINITY;
    double speed = 0;
    if (i < 2) {
      position = 0;
      speed = 3;
    }
    wheels[i]->setPosition(position);
    wheels[i]->setVelocity(speed);
  }

  // reset the robot to its initial position
  // cout << "Setting translation to " << initial_position[0] << ", " << initial_position[1] << ", " << initial_position[2] << endl;
  // cout << "Setting rotation to " << initial_rotation[0] << ", " << initial_rotation[1] << ", " << initial_rotation[2] << ", " << initial_rotation[3] << endl;

  double *position = (double*) malloc(3 * sizeof(double));
  double *rotation = (double*) malloc(4 * sizeof(double));
  position[0] = initial_position[0];
  position[1] = initial_position[1];
  position[2] = initial_position[2];
  rotation[0] = initial_rotation[0];
  rotation[1] = initial_rotation[1];
  rotation[2] = initial_rotation[2];
  rotation[3] = initial_rotation[3];
  
  translation_field->setSFVec3f(position);
  rotation_field->setSFRotation(rotation);

  free(position);
  free(rotation);
}

int main(int argc, char **argv) {
  robot = new Supervisor();
  Node *robot_node = robot->getFromDef("ROBOT");

  // get the initial position and rotation of the robot
  Field *translation_field = robot_node->getField("translation");
  Field *rotation_field = robot_node->getField("rotation");
  
  const double *position = translation_field->getSFVec3f();
  const double *rotation = rotation_field->getSFRotation();

  array<double, 3> initial_position{position[0], position[1], position[2]};
  array<double, 4> initial_rotation{rotation[0], rotation[1], rotation[2], rotation[3]};
    
  camera = robot->getCamera("camera");
  camera->enable(SAMPLING_PERIOD);  

  array<double, 2> speed_multipliers{5, 0};
  array<double, 2> steering_multipliers{1, 0};

  for (int i = 1; i < 10; i++) {
    for (int j = 1; j < 10; j++) {
      speed_multipliers[0] = i;
      
      steering_multipliers[0] = j;
    
      reset_robot(translation_field, rotation_field, initial_position, initial_rotation);
      int fitness = run_robot(
        translation_field,
        speed_multipliers,
        steering_multipliers
      );
      cout << "Fitness: " << fitness << endl;
    }
  }

  delete robot;

  return 0;
}
