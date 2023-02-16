"""csci3302_lab2 controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import math
from controller import Robot, Motor, DistanceSensor
import os

# Ground Sensor Measurements under this threshold are black
# measurements above this threshold can be considered white.
# TODO: Fill this in with a reasonable threshold that separates "line detected" from "no line detected"
# when the gs detects the line, the signal drops to arounnd 300, so 350 is an appropriate threshold
GROUND_SENSOR_THRESHOLD = 350

# These are your pose values that you will update by solving the odometry equations
pose_x = 0
pose_y = 0
pose_theta = 0

# Index into ground_sensors and ground_sensor_readings for each of the 3 onboard sensors.
LEFT_IDX = 2
CENTER_IDX = 1
RIGHT_IDX = 0

# create the Robot instance.
robot = Robot()

# ePuck Constants
EPUCK_AXLE_DIAMETER = 0.053  # ePuck's wheels are 53mm apart.
# TODO: To be filled in with ePuck wheel speed in m/s
EPUCK_MAX_WHEEL_SPEED = 0.1163
MAX_SPEED = 6.28

# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# Initialize Motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Initialize and Enable the Ground Sensors
gsr = [0, 0, 0]
ground_sensors = [robot.getDevice('gs0'), robot.getDevice(
    'gs1'), robot.getDevice('gs2')]
for gs in ground_sensors:
    gs.enable(SIM_TIMESTEP)

# Allow sensors to properly initialize
for i in range(10):
    robot.step(SIM_TIMESTEP)

vL = 0.5*MAX_SPEED  # TODO: Initialize variable for left speed
vR = 0.5*MAX_SPEED  # TODO: Initialize variable for right speed

# Main Control Loop:
while robot.step(SIM_TIMESTEP) != -1:

    # Read ground sensor values
    for i, gs in enumerate(ground_sensors):
        gsr[i] = gs.getValue()

    # print(gsr)  # TODO: Uncomment to see the ground sensor values!

    # Hints:
    #
    # 1) Setting vL=MAX_SPEED and vR=-MAX_SPEED lets the robot turn
    # right on the spot. vL=MAX_SPEED and vR=0.5*MAX_SPEED lets the
    # robot drive a right curve.
    #
    # 2) If your robot "overshoots", turn slower.
    #
    # 3) Only set the wheel speeds once so that you can use the speed
    # that you calculated in your odometry calculation.
    #
    # 4) Disable all console output to simulate the robot superfast
    # and test the robustness of your approach.
    #

    # Implement control code to cause the ePuck to follow the
    # black line on the ground within the “line_follower” state.
    # You should use +/- <motor>.getMaxVelocity() for motor velocity values or slower.

    # a) If the center ground sensor detects the line, the robot should drive forward.
    # b) If the left ground sensor detects the line, the robot should rotate counterclockwise in place.
    # c) If the right ground sensor detects the line, the robot should rotate clockwise in place.
    # d) Otherwise, if none of the ground sensors detect the line, rotate counterclockwise in place.
    # (This will help the robot re-find the line when it overshoots on corners!)

    # TODO: Insert Line Following Code Here

    # Store the values of the sensors
    gs_left = gsr[0]
    gs_center = gsr[1]
    gs_right = gsr[2]

    # print("gs_left:[%5f], gs_center:[%5f], gs_right:[%5f]." % (gs_left, gs_center, gs_right))

    left_line = gsr[0] < GROUND_SENSOR_THRESHOLD
    center_line = gsr[1] < GROUND_SENSOR_THRESHOLD
    right_line = gsr[2] < GROUND_SENSOR_THRESHOLD

    # d) Otherwise, if none of the ground sensors detect the line, rotate counterclockwise in place.
    if not left_line and not center_line and not right_line:
        print("---> No Line:")
        print("--->gs_left:[%5f], gs_center:[%5f], gs_right:[%5f]." %
              (gs_left, gs_center, gs_right))
        vR = 0.4*MAX_SPEED
        vL = -0.4*MAX_SPEED

    # a) If the center ground sensor detects the line, the robot should drive forward.
    # abs(gs_center - gs_left) > 1
    if (center_line and left_line and right_line):
        vR = 0.5*MAX_SPEED
        vL = 0.5*MAX_SPEED
        print(" Three Sensors:")
        print(" gs_left:[%5f], gs_center:[%5f], gs_right:[%5f]." %
              (gs_left, gs_center, gs_right))

    # b) If the left ground sensor detects the line, the robot should rotate counterclockwise in place.
    # account for the noise:  abs(gs_center - gs_left) > 0.5
    elif ((gs_left == min(gsr) or left_line or (center_line and left_line)) and
          abs(gs_center - gs_left) > 0.5
          ):

        vR = 0.3*MAX_SPEED
        vL = -0.3*MAX_SPEED
        print("-> Left Line:")
        print("-> gs_left:[%5f], gs_center:[%5f], gs_right:[%5f]." %
              (gs_left, gs_center, gs_right))

    # c) If the right ground sensor detects the line, the robot should rotate clockwise in place.
    # account for the noise: abs(gs_center - gs_left) > 0.5
    elif (
        (gs_right == min(gsr) or right_line or (center_line and right_line)) and
        abs(gs_center - gs_left) > 0.5
    ):
        vR = -0.3*MAX_SPEED
        vL = 0.3*MAX_SPEED
        print("--> Right Line:")
        print("--> gs_left:[%5f], gs_center:[%5f], gs_right:[%5f]." %
              (gs_left, gs_center, gs_right))

    elif gs_center == min(gsr) or center_line:
        vR = 0.5*MAX_SPEED
        vL = 0.5*MAX_SPEED
        print("> Center Line:")
        print("> gs_left:[%5f], gs_center:[%5f], gs_right:[%5f]." %
              (gs_left, gs_center, gs_right))

    # TODO: Call update_odometry Here

    # Hints:
    #
    # 1) Divide vL/vR by MAX_SPEED to normalize, then multiply with
    # the robot's maximum speed in meters per second.
    #
    # 2) SIM_TIMESTEP tells you the elapsed time per step. You need
    # to divide by 1000.0 to convert it to seconds
    #
    # 3) Do simple sanity checks. In the beginning, only one value
    # changes. Once you do a right turn, this value should be constant.
    #
    # 4) Focus on getting things generally right first, then worry
    # about calculating odometry in the world coordinate system of the
    # Webots simulator first (x points down, y points right)

    # TODO: Insert Loop Closure Code Here

    # Hints:
    #
    # 1) Set a flag whenever you encounter the line
    #
    # 2) Use the pose when you encounter the line last
    # for best results

    # print("Current pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta))
    leftMotor.setVelocity(vL)
    rightMotor.setVelocity(vR)
