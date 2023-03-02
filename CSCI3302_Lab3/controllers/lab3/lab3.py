"""lab3 controller."""
# Copyright University of Colorado Boulder 2022
# CSCI 3302 "Introduction to Robotics" Lab 3 Base Code.

from controller import Robot, Motor
import math

# TODO: Fill out with correct values from Robot Spec Sheet (or inspect PROTO definition for the robot)
MAX_SPEED = 6.67  # [rad/s]
MAX_SPEED_MS = 0.22  # [m/s]
AXLE_LENGTH = 0.178  # [m]

MOTOR_LEFT = 0  # Left wheel index
MOTOR_RIGHT = 1  # Right wheel index


# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
timestep_s = timestep / 10000

# The Turtlebot robot has two motors
part_names = ("left wheel motor", "right wheel motor")


# Set wheels to velocity control by setting target position to 'inf'
# You should not use target_pos for storing waypoints. Leave it unmodified and
# use your own variable to store waypoints leading up to the goal
target_pos = ('inf', 'inf')
robot_parts = []

for i in range(len(part_names)):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))

# Get the maximum linear and angular velocities for the wheels
max_linear_velocity = min(
    robot_parts[0].getMaxVelocity(), robot_parts[1].getMaxVelocity())
# print("-> max_linear_velocity for wheels: ", max_linear_velocity)

# Odometry
pose_x = 0
pose_y = 0
pose_theta = 0

# Rotational Motor Velocity [rad/s]
vL = 0
vR = 0

# TODO
# Create you state and goals (waypoints) variable here
# You have to MANUALLY figure out the waypoints, one sample is provided for you in the instructions
# Waypoints, array of (x,y) coordinates to traverse
# Fill with correct values!!

waypoints = [[-8, -5],
             [-6, -6],
             [-4.5, -4],
             [-3.3, -3.8],
             [-1.7, -2],
             [-1.7, -1.7]]

i, j = 0, 0


# Starting point on map:
# x: -8 m
# y: -5 m
# rad: 0.523599 rad (z)

# waypoint 1:
# x: -6 m
# y: -6 m
# rad: 0.523599 rad (z)


while robot.step(timestep) != -1:

    # STEP 2.1: Calculate error with respect to current and goal position

    # goal_x = waypoints[i][j]
    # goal_y = waypoints[i][j+1]

    goal_x = waypoints[i+1][j] - waypoints[i][j]
    goal_y = waypoints[i+1][j+1] - waypoints[i][j+1]

    # print("goal_x, goal_y", goal_x, " ", goal_y)
    goal_theta = 0.5

    ## Position Error : rho ##
    # sqrt( (xr-xg)^2 + (yr - yg)^2)
    rho = math.sqrt((pose_x - goal_x)**2 + (pose_y - goal_y)**2)
    # print("-> position error:", rho)

    ## Bearing Error: alpha##
    alpha = math.atan2((goal_y-pose_y), (goal_x-pose_x))-pose_theta
    # print("-> Bearing Error:", alpha)

    ## Heading Error: nu ##
    nu = goal_theta - pose_theta
    # print("-> Heading Error:", alpha)

    # STEP 2.2: Feedback Controller

    # STEP 1: Inverse Kinematics Equations (vL and vR as a function dX and dTheta)
    # Note that vL and vR in code is phi_l and phi_r on the slides/lecture
    xR_dot = 0
    thetaR_dot = 0
    radius = (MAX_SPEED_MS/MAX_SPEED)
    d = AXLE_LENGTH

    # STEP 2.4: Clamp wheel speeds
    # controller gains p1, p2, p3
    # xR_dot = p1 * rho
    # thetaR_dot = p2 * alpha + p3 * nu

    # need conditional logic to determine controller gains
    # Prioritize BEaring error
    p1, p2, p3 = 0, 0, 0
    if (abs(alpha) > 0.5):
        p1 = 2  # small gains for rho
        p2 = 4  # higher gains for alpha
        p3 = 0  # no update for nu
    else:
        # Prioratize position error
        if (abs(rho) > 0.5):
            p1 = 4  # higher gains for rho i.e. position
            p2 = 2
            p3 = 0
        else:
            # Prioritize heading error
            p1 = 2
            p2 = 2
            p3 = 4

    xR_dot = p1 * rho
    thetaR_dot = p2 * alpha + p3 * nu

    # calculate xR_dot and thetaR_dot
    # xR_dot = vL/2 + vR/2
    # thetaR_dot = vR/d - vL/d

    # STEP 2.3: Proportional velocities
    vL = xR_dot - (thetaR_dot/2)*d  # Left wheel velocity in rad/s
    vR = xR_dot + (thetaR_dot/2)*d

    # 2.7 clamp the velocities if they exceed MAX_SPEED
    if vL > MAX_SPEED:
        vL = MAX_SPEED
    if vR > MAX_SPEED:
        vR = MAX_SPEED

    # print("vR after updating: ", vR)
    # print("vL after updating: ", vL)

    # TODO
    # Use Your Lab 2 Odometry code after these 2 comments. We will supply you with our code next week
    # after the Lab 2 deadline but you free to use your own code if you are sure about its correctness

    # NOTE that the odometry should ONLY be a function of
    # (vL, vR, MAX_SPEED, MAX_SPEED_MS, timestep, AXLE_LENGTH, pose_x, pose_y, pose_theta)
    # Odometry code. Don't change speeds (vL and vR) after this line

    # Rx_dot = vR/2 + vL/2 #Forward speed of the robot
    # Rw_dot = vR/d - vL/d #Net rotation in the local frame

    # ix_dot = Rx_dot * math.cos(pose_theta) #Robots x_dot contribution in i coordinate frame
    # iy_dot = Rx_dot * math.sin(pose_theta) #Robots y_dot contribution in i coordinate frame
    # itheta_dot = Rw_dot #Robots theta_dot contribution in i coordinate frame

    # if (vR*vL < 0):
    #     pose_theta += itheta_dot*timestep_s #multiply by change in t to find rotation and add to previous value
    # #If the robot is moving straight, update its x and y values
    # else:
    #     pose_x += ix_dot*timestep_s #multiply by change in t to find displacement and add to previous value
    #     pose_y += iy_dot*timestep_s #multiply by change in t to find displacement and add to previous value

    # Odometry code from Piazza
    distL = vL/MAX_SPEED * MAX_SPEED_MS * timestep/1000.0

    distR = vR/MAX_SPEED * MAX_SPEED_MS * timestep/1000.0

    pose_x += (distL+distR) / 2.0 * math.cos(pose_theta)

    pose_y += (distL+distR) / 2.0 * math.sin(pose_theta)

    pose_theta += (distR-distL)/AXLE_LENGTH

    # Debugging Code
    print(f'Pose {pose_x=} {pose_y=} {pose_theta=}')

    print(f'Goal {goal_x=} {goal_y=} {goal_theta=}')

    print(f'Error values {rho=} {alpha=} {nu=}')

    print(f'Speeds {vR=} {vL=}')
    print("---------------------------------")

    ########## End Odometry Code ##################

    ########## Do not change ######################
    # Bound pose_theta between [-pi, 2pi+pi/2]
    # Important to not allow big fluctuations between timesteps (e.g., going from -pi to pi)
    if pose_theta > 6.28+3.14/2:
        pose_theta -= 6.28
    if pose_theta < -3.14:
        pose_theta += 6.28
    ###############################################

    # TODO
    # Set robot motors to the desired velocities
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
