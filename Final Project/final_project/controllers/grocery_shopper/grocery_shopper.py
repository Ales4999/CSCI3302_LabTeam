"""grocery controller."""

# Nov 2, 2022
# Group: WALL-E's
# Team Members:
# Mark Abbott
# Miles Sanders
# Alberto Espinosa

from controller import Robot, Keyboard
from controller import CameraRecognitionObject
import math
import random
import pdb
import pickle
import copy
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
# Initialization
color_ranges = []

print("=== Initializing Grocery Shopper...")
# Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633  # [m/s]
AXLE_LENGTH = 0.4044  # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5  # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint", "gripper_right_finger_joint")

#

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27,
              1.32, 0.0, 1.41, 'inf', 'inf', 0.045, 0.045)

robot_parts = {}
for i, part_name in enumerate(part_names):
    robot_parts[part_name] = robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(
        robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc = robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc = robot.getDevice("gripper_right_finger_joint_sensor")
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

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Odometry
pose_x = 0
pose_y = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = []  # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE /
                            2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
# Only keep lidar readings not blocked by robot chassis
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83]

map = None

mode = 'manual'
# mode = 'planner'
# mode = 'autonomous'

map = np.zeros(shape=[360, 192])
# map = np.zeros(shape=[372, 372])

waypoints = []


# ------------------------------------------------------------------
# Planning Algorithm   ---- Planner ----
# Tier 1: Teleoperation: use the keyboard to direct robot to waypoints
# Tier 2: A*
# Tier 3: RRT

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


class RRT:
    def __init__(self, start, goal, obstacle_map, waypoints):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacle_map = obstacle_map
        self.width = obstacle_map.shape[1]
        self.height = obstacle_map.shape[0]
        self.nodes = [self.start]
        self.waypoints = waypoints
        self.path = []
        self.epsilon = 10
        self.goal_tolerance = 10

    def plan(self):
        for waypoint in self.waypoints:
            for i in range(1000):
                if random.random() < 0.05:
                    x, y = self.goal.x, self.goal.y
                else:
                    x, y = self.generate_random_point()

                nearest_node = self.get_nearest_node(x, y)

                new_node = self.steer(nearest_node, x, y)

                if self.obstacle_free(nearest_node, new_node):
                    self.nodes.append(new_node)
                    new_node.parent = nearest_node

                if self.goal_reached(new_node, self.goal):
                    self.path = self.get_path(new_node)
                    break

            if self.path:
                break

        return self.path

    def generate_random_point(self):
        x = random.randint(0, self.width-1)
        y = random.randint(0, self.height-1)
        return x, y

    def get_nearest_node(self, x, y):
        distances = [math.sqrt((node.x-x)**2 + (node.y-y)**2)
                     for node in self.nodes]
        nearest_index = np.argmin(distances)
        return self.nodes[nearest_index]

    def steer(self, nearest_node, x, y):
        dx = x - nearest_node.x
        dy = y - nearest_node.y
        theta = math.atan2(dy, dx)
        new_x = nearest_node.x + self.epsilon * math.cos(theta)
        new_y = nearest_node.y + self.epsilon * math.sin(theta)
        return Node(int(new_x), int(new_y))

    def obstacle_free(self, node1, node2):
        x1, y1, x2, y2 = node1.x, node1.y, node2.x, node2.y

        for i in range(0, 101, 10):
            x = int(i * x2 + (100-i) * x1) // 100
            y = int(i * y2 + (100-i) * y1) // 100
            if self.obstacle_map[y][x] == 1:
                return False

        return True

    def goal_reached(self, node, goal):
        return math.sqrt((node.x-goal.x)**2 + (node.y-goal.y)**2) < self.goal_tolerance

    def get_path(self, node):
        path = []
        current = node
        while current:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]


if mode == 'planner':
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    start_x = 0
    start_y = 0
    start_w = (start_x,  start_y)  # (Pose_X, Pose_Y) in meters

    # End coordinates
    end_x = 0
    end_y = 0
    end_w = (end_x, end_y)  # (Pose_X, Pose_Y) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    # (x, y) in 360x360 map
    start = (abs(int(start_w[0]*30)), abs(int(start_w[1]*30)))
    # (x, y) in 360x360 map
    end = (abs(int(end_w[0]*30)), abs(int(end_w[1]*30)))

    # Part 2.1: Load map (map.npy) from disk and visualize it
    map = np.load('./map.npy')

    # Part 2.2: Compute an approximation of the “configuration space”
    # define quadratic kernel
    kernel_size = 22  # had to guess and check an appropriate value
    kernel = np.ones((kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)  # normalize kernel to sum to 1

    # convolve binary occupancy grid map with kernel
    configuration_space = convolve2d(map, kernel, mode='same')
    configuration_space = configuration_space * 360  # scale

    # threshold configuration space to create binary configuration space
    configuration_space[configuration_space < 1] = 0
    configuration_space[configuration_space >= 1] = 1

    # save config file
    np.save('config_space.npy', configuration_space)
    # load config file
    config_space = np.load('./config_space.npy')

    # Create a figure
    fig, ax = plt.subplots()
    # Plot the configuration space
    ax.imshow(config_space, cmap='binary')
    # Show the plot
    # plt.show()

    # Define the waypoints as a list of (x, y) tuples
    waypoints = [(50, 50), (100, 100), (150, 150)]

    # Create a new RRT planner
    planner = RRT(start, end, config_space, waypoints)

    # Plan a path
    path = planner.plan()

    # Plot the path and the obstacles
    plt.imshow(config_space, cmap='gray', origin='lower')
    plt.plot(start[1], start[0], 'go')
    plt.plot(end[1], end[0], 'ro')
    for waypoint in waypoints:
        plt.plot(waypoint[1], waypoint[0], 'yo')
    if path:
        x, y = zip(*path)
        plt.plot(y, x, '-b')
    plt.show()

    # convert back to world coordinates
    # (abs(int(start_w[0]*30)), abs(int(start_w[1]*30)))
    path_w = []
    for node in path:
        x = (abs(int(node[0]/30)))
        y = (abs(int(node[1]/30)))
        # create a node for the world coordinate system
        node_w = (x, y)
        # paint the world path
        ax.scatter(x, y)
        path_w.append(node_w)

    # save the world coordinate waypoint to disk
    np.save("path.npy", path_w)


# ------------------------------------------------------------------
# Helper Functions
def add_color_range_to_detect(lower_bound, upper_bound):
  '''
  @param lower_bound: Tuple of BGR values
  @param upper_bound: Tuple of BGR values
  '''
  global color_ranges
  color_ranges.append([lower_bound, upper_bound])

def check_if_color_in_range(bgr_tuple):
  '''
  @param bgr_tuple: Tuple of BGR values
  @returns Boolean: True if bgr_tuple is in any of the color ranges specified in color_ranges
  '''
  global color_ranges
  for entry in color_ranges:
    lower, upper = entry[0], entry[1]
    in_range = True
    for i in range(len(bgr_tuple)):
      if bgr_tuple[i] < lower[i] or bgr_tuple[i] > upper[i]:
        in_range = False
        break
    if in_range: return True
  return False

def do_color_filtering(img):
  img_height = img.shape[0]
  img_width = img.shape[1]
  # Create a matrix of dimensions [height, width] using numpy
  mask = np.zeros([img_height, img_width]) # Index mask as [height, width] (e.g.,: mask[y,x])
  for i in range(img_height):
    for j in range(img_width):
      # print(img[i,j])
      if check_if_color_in_range(img[i,j]) is True:
        mask[i,j] = 1
  return mask

def expand(img_mask, cur_coordinate, coordinates_in_blob):
  if cur_coordinate[0] < 0 or cur_coordinate[1] < 0 or cur_coordinate[0] >= img_mask.shape[0] or cur_coordinate[1] >= img_mask.shape[1]:
    return
  if img_mask[cur_coordinate[0], cur_coordinate[1]] == 0.0:
    return

  img_mask[cur_coordinate[0],cur_coordinate[1]] = 0
  coordinates_in_blob.append(cur_coordinate)

  above = [cur_coordinate[0]-1, cur_coordinate[1]]
  below = [cur_coordinate[0]+1, cur_coordinate[1]]
  left = [cur_coordinate[0], cur_coordinate[1]-1]
  right = [cur_coordinate[0], cur_coordinate[1]+1]
  for coord in [above, below, left, right]: 
    expand(img_mask, coord, coordinates_in_blob)

def expand_nr(img_mask, cur_coord, coordinates_in_blob):
  coordinates_in_blob = []
  coordinate_list = [cur_coord] # List of all coordinates to try expanding to
  while len(coordinate_list) > 0:
    cur_coordinate = coordinate_list.pop() # Take the first coordinate in the list and perform 'expand' on it
    if cur_coordinate[0] < 0 or cur_coordinate[1] < 0 or cur_coordinate[0] >= img_mask.shape[0] or cur_coordinate[1] >= img_mask.shape[1]:
        continue
    if img_mask[cur_coordinate[0], cur_coordinate[1]] == 0.0 or cur_coordinate in coordinates_in_blob: 
        continue
    img_mask[cur_coordinate[0], cur_coordinate[1]] = 0
    coordinates_in_blob.append(cur_coordinate)
    above = (cur_coordinate[0]-1, cur_coordinate[1])
    below = (cur_coordinate[0]+1, cur_coordinate[1])
    left = (cur_coordinate[0], cur_coordinate[1]-1)
    right = (cur_coordinate[0], cur_coordinate[1]+1)
    for coord in [above, below, left, right]:  
      coordinate_list.append(coord)
  return coordinates_in_blob

def get_blobs(img_mask):
  img_mask_height = img_mask.shape[0]
  img_mask_width = img_mask.shape[1]
  mask_copy = copy.copy(img_mask)
  blobs_list = [] # List of all blobs, each element being a list of coordinates belonging to each blob
  for i in range(img_mask_height):
    for j in range(img_mask_width):
      if img_mask[i,j] ==  1:
        coord = [i,j]
        blob_coords = []
        blob_expand = expand_nr(mask_copy,coord,blob_coords)
        blobs_list.append(blob_expand)
  return blobs_list

def identify_cube():
   img = camera.getImageArray() #Read image from current frame of robot camera
   img = np.asarray(img, dtype=np.uint8)
   add_color_range_to_detect([158,167,0], [222,222,73]) # Detect yellow
    # Create img_mask of all foreground pixels, where foreground is defined as passing the color filter. Same function as homework 3
   img_mask = do_color_filtering(img)
   # Find all the blobs in the img_mask. Same function as homework 3
   blobs = get_blobs(img_mask)
   #Print location of detected object using Webots API
   if len(blobs)>0:
       obj = camera.getRecognitionObjects()[0]
       position = obj.getPosition()
       print("Cube Location In Relation To Camera:")
       print(position[0],position[1])
   return

gripper_status = "closed"

# Main Loop
while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    # Ground truth pose
    pose_x = 15 - gps.getValues()[0]
    pose_y = 8 - gps.getValues()[1]

    n = compass.getValues()
    rad = -((math.atan2(n[1], n[0]))-1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi

        # Convert detection from robot coordinates into world coordinates
        wx = math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy = math.sin(t)*rx + math.cos(t)*ry + pose_y
        #print(wx, wy)

        ################ ^ [End] Do not modify ^ ##################

        # print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f, x: %f, y: %f" % (rho,alpha,rx,ry,wx,wy,x,y))
        #print(wx,wy)
        if wx >= 30:
            wx = 29.999
        if wy >= 16:
            wy = 15.999
        if rho < LIDAR_SENSOR_MAX_RANGE:

            # ---- Part 1.3: visualize map gray values. ----
            x = abs(int(wx*12))
            y = abs(int(wy*12))

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
            #print(x,y)
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
            display.drawPixel(y, x)

    # draw the robots line
    display.setColor(int(0xFF0000))
    display.drawPixel(abs(int(pose_y*12)),abs(int(pose_x*12)))

    if mode == 'manual':
        key = keyboard.getKey()
        if key == ord('C'):
            identify_cube()
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

        elif key == ord('L'):
            map = np.load("map.npy")
            plt.imshow(np.fliplr(map))
            plt.title('Map')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
            print("Map loaded...")
        else:  # slow down
            vL *= 0.75
            vR *= 0.75

    robot_parts["wheel_left_joint"].setVelocity(0.5*vL)
    robot_parts["wheel_right_joint"].setVelocity(0.5*vR)

    if (gripper_status == "open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue() <= 0.005:
            gripper_status = "closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue() >= 0.044:
            gripper_status = "open"
