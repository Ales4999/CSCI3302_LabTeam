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
GROUND_SENSOR_THRESHOLD = 600

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
EPUCK_MAX_WHEEL_SPEED = 0.1163 #calculated using ds/dt
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
ground_sensors = [robot.getDevice('gs0'), robot.getDevice('gs1'), robot.getDevice('gs2')]
for gs in ground_sensors:
    gs.enable(SIM_TIMESTEP)

# Allow sensors to properly initialize
for i in range(10):
    robot.step(SIM_TIMESTEP)

vL = 0.5*MAX_SPEED  # TODO: Initialize variable for left speed
vR = 0.5*MAX_SPEED  # TODO: Initialize variable for right speed

wheelRadius = EPUCK_MAX_WHEEL_SPEED/MAX_SPEED
MAX_SPEED_MS = ((vL/vR)/MAX_SPEED)*EPUCK_MAX_WHEEL_SPEED
SIM_TIMESTEP_S = SIM_TIMESTEP/1000.0
start_counter = 0

# Main Control Loop:
while robot.step(SIM_TIMESTEP) != -1:
    
    # Read ground sensor values
    for i, gs in enumerate(ground_sensors):
        gsr[i] = gs.getValue()

    # print(gsr)  # TODO: Uncomment to see the ground sensor values!

    # Store the values of the sensors
    gs_left = gsr[0]
    gs_center = gsr[1]
    gs_right = gsr[2]

    #Name the value of the sensors and check if they are below the threshold
    left_line = gsr[0] < GROUND_SENSOR_THRESHOLD
    center_line = gsr[1] < GROUND_SENSOR_THRESHOLD
    right_line = gsr[2] < GROUND_SENSOR_THRESHOLD

    # TODO: Insert Line Following Code Here
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

    # d) Otherwise, if none of the ground sensors detect the line, rotate counterclockwise in place.
    if (not left_line and not center_line and not right_line):
        start_counter = 0
        vR = 0.3*MAX_SPEED
        vL = -0.3*MAX_SPEED
    
    # a) If the center ground sensor detects the line, the robot should drive forward.
    elif ((center_line and not left_line and not right_line) or 
    (center_line and left_line and not right_line) or 
    (center_line and right_line and not left_line)):
    
        start_counter = 0
        vR = 0.5*MAX_SPEED
        vL = 0.5*MAX_SPEED

    # b) If the left ground sensor detects the line, the robot should rotate counterclockwise in place.
    elif (left_line and not center_line and not right_line):
        start_counter = 0
        vR = 0.3*MAX_SPEED
        vL = -0.3*MAX_SPEED

    # c) If the right ground sensor detects the line, the robot should rotate clockwise in place.
    elif (right_line and not center_line and not left_line):
        start_counter = 0
        vR = -0.3*MAX_SPEED
        vL = 0.3*MAX_SPEED

    #If the robot is crossing the start line or sometimes all three get triggered in the corners
    elif (center_line and right_line and left_line):
        start_counter += 1
        vR = 0.5*MAX_SPEED
        vL = 0.5*MAX_SPEED

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

    Rx_dot = (vR*wheelRadius)/2 + (vL*wheelRadius)/2 #Forward speed of the robot
    Rw_dot = (vR*wheelRadius)/EPUCK_AXLE_DIAMETER - (vL*wheelRadius)/EPUCK_AXLE_DIAMETER #Net rotation in the local frame

    ix_dot = Rx_dot * math.cos(pose_theta) #Robots x_dot contribution in i coordinate frame
    iy_dot = Rx_dot * math.sin(pose_theta) #Robots y_dot contribution in i coordinate frame
    itheta_dot = Rw_dot #Robots theta_dot contribution in i coordinate frame

    # TODO: Insert Loop Closure Code Here
    # Hints:
    # 1) Set a flag whenever you encounter the line
    # 2) Use the pose when you encounter the line last
    # for best results

    #The start_counter reaches about 10 after it passes over the start line, hence, reset its coordinates to (0,0,0)
    if (start_counter > 8):
        pose_x = 0
        pose_y = 0
        pose_theta = 0
    #If the robot is rotating on the spot, update its pose value
    elif (vR*vL < 0):
        pose_theta += itheta_dot*SIM_TIMESTEP_S #multiply by change in t to find rotation and add to previous value
    #If the robot is moving straight, update its x and y values
    else:
        pose_x += ix_dot*SIM_TIMESTEP_S #multiply by change in t to find displacement and add to previous value
        pose_y += iy_dot*SIM_TIMESTEP_S #multiply by change in t to find displacement and add to previous value

    print("Current pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta))
    #Set the motor velocities with the new vL and vR rotational speeds
    leftMotor.setVelocity(vL)
    rightMotor.setVelocity(vR)
