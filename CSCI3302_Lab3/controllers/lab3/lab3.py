"""lab3 controller."""
# Copyright University of Colorado Boulder 2022
# CSCI 3302 "Introduction to Robotics" Lab 3 Base Code.

from controller import Robot, Motor
import math

# TODO: Fill out with correct values from Robot Spec Sheet (or inspect PROTO definition for the robot)
MAX_SPEED = 6.67 # [rad/s]
MAX_SPEED_MS = 0.22 # [m/s]
AXLE_LENGTH = 0.178 # [m]


MOTOR_LEFT = 0 # Left wheel index
MOTOR_RIGHT = 1 # Right wheel index

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

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
max_linear_velocity = min( robot_parts[0].getMaxVelocity(), robot_parts[1].getMaxVelocity())
#print("-> max_linear_velocity for wheels: ", max_linear_velocity)

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

# Rotational Motor Velocity [rad/s]
vL = 0
vR = 0

#Waypoints, array of (x,y) coordinates to traverse 
#Fill with correct values!!
waypoints  = [[0, 1],
              [2, 3],
              [4, 5],
              [6, 7],
              [8, 9],
              [10,11]]

# TODO
# Create you state and goals (waypoints) variable here
# You have to MANUALLY figure out the waypoints, one sample is provided for you in the instructions

while robot.step(timestep) != -1:

    # STEP 2.1: Calculate error with respect to current and goal position
    
    #Prof said in class odometry was going to be given  
    #down the line this must be updated somehow
    goal_x = waypoints[0][0]
    goal_y = waypoints[0][1] 
    
    # sqrt( (xr-xg)^2 + (yr - yg)^2) 
    rho = math.sqrt( (pose_x - goal_x)^2  + (pose_y - goal_y )^2)
    pass   
    
    # STEP 2.2: Feedback Controller
    pass
    xR_dot = 0
    thetaR_dot = 0
    radius =  (MAX_SPEED_MS/MAX_SPEED)
    d = AXLE_LENGTH
    # STEP 1: Inverse Kinematics Equations (vL and vR as a function dX and dTheta)
    # Note that vL and vR in code is phi_l and phi_r on the slides/lecture
    vL = xR_dot - (thetaR_dot/2)*d
    vR = xR_dot + (thetaR_dot/2)*d

    xR_dot =(vL/2 + vR/2)
    thetaR_dot = (vR/d - vL/d)

    pass
    
    # STEP 2.3: Proportional velocities
    vL = 0 # Left wheel velocity in rad/s
    vR = 0 # Right wheel velocity in rad/s
    pass

    # STEP 2.4: Clamp wheel speeds
    pass


    
    # TODO
    # Use Your Lab 2 Odometry code after these 2 comments. We will supply you with our code next week 
    # after the Lab 2 deadline but you free to use your own code if you are sure about its correctness
    
    # NOTE that the odometry should ONLY be a function of 
    # (vL, vR, MAX_SPEED, MAX_SPEED_MS, timestep, AXLE_LENGTH, pose_x, pose_y, pose_theta)
    # Odometry code. Don't change speeds (vL and vR) after this line
    
    
    

    ########## End Odometry Code ##################
    
    ########## Do not change ######################
    # Bound pose_theta between [-pi, 2pi+pi/2]
    # Important to not allow big fluctuations between timesteps (e.g., going from -pi to pi)
    if pose_theta > 6.28+3.14/2: pose_theta -= 6.28
    if pose_theta < -3.14: pose_theta += 6.28
    ###############################################

    # TODO
    # Set robot motors to the desired velocities
    robot_parts[MOTOR_LEFT].setVelocity(0)
    robot_parts[MOTOR_RIGHT].setVelocity(0)

    