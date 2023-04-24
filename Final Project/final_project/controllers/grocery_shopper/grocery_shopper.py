"""grocery controller."""

# Nov 2, 2022

from controller import Robot, Keyboard
import math
import numpy as np

#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None

mode = 'manual'
#mode = 'autonomous'

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

map = np.zeros(shape=[360, 360])
# map = np.zeros(shape=[372, 372])

waypoints = []


# ------------------------------------------------------------------
# Helper Functions


gripper_status="closed"

# Main Loop
while robot.step(timestep) != -1:

    # Ground truth pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]

    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(
        lidar_sensor_readings)-83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.

        # Convert detection from robot coordinates into world coordinates
        wx = math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy = math.sin(t)*rx + math.cos(t)*ry + pose_y

        ################ ^ [End] Do not modify ^ ##################

        # print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f, x: %f, y: %f" % (rho,alpha,rx,ry,wx,wy,x,y))

        if wx <= -12:
            wx = -11.999
        if wy <= -12:
            wy = -11.999
        if rho < LIDAR_SENSOR_MAX_RANGE:

            # ---- Part 1.3: visualize map gray values. ----
            x = abs(int(wx*30))
            y = abs(int(wy*30))

            # if x >= 360:
            # continue
            # if y >= 360 :
            # continue

            # scale = 300
            # display.setColor(0xFF0000)  # red
            # display_x = round(pose_x * scale)
            # display_y = round(pose_y * scale)

            increment_value = 5e-3

            # need to bound increment value?
            # or index?
            map[x, y] += increment_value

            # check what is getting stored in the map
            # print(map[x][y])

            # make sure the value does not exceed 1
            # map = np.clip(map, None, 1)  # Keep values within [0,1]
            # You will eventually REPLACE the following lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.

            # draw map on diplsay
            # convert the gray scale
            # set g to a vallue depending on our map
            # g_map = min( map[x][y], 1.0)
            g = min(map[x][y], 1.0)

            color = (g*256**2+g*256+g)*255

            display.setColor(int(color))
            display.drawPixel(x, y)


    # draw the robots line
    display.setColor(int(0xFF0000))
    display.drawPixel(abs(int(pose_x*30)), abs(int(pose_y*30)))
    
    if mode == 'manual':
        key = keyboard.getKey()
        while (keyboard.getKey() != -1):
            pass
        if key == keyboard.LEFT:
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            threshold_value = 0.5
            thresholded_map = np.multiply(map > threshold_value, 1)
            map_name = 'map.npy'

            # save the thresholded map data to a file
            map_trimmed = thresholded_map[0:800, 0:360]
            np.save(map_name, map_trimmed)

            print("Map file saved as %s" % (map_name))

    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            gripper_status="closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            gripper_status="open"
