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

part_positions = target_pos

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

# Initialization of color changes
color_ranges = []

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

# mode = 'manual'
# mode = 'autonomous'
# mode = 'SLAM'
# mode = 'planner'
mode = 'path_following'

arm_mode = 'manual'
# arm_mode = 'teleoperation_IK'

arm_positioning = 'on'
# arm_positioning = 'off'

map = np.zeros(shape=[360, 192])
# map = np.zeros(shape=[372, 372])

# waypoints = [(11.20, 1.55), (18.26, 5.05),
#              (17.74, 7.00), (13.97, 7.24), (12.51, 5.38), (11.28, 5.29),
#              (9.34, 10.73), (12.34, 10.73), (17.28, 13.50)]
wall_counter = 0
beginning = 0
stage_counter = 0
state = 0
path_follow = False
path_i = 0

while robot.step(timestep) != -1:
    initial_heading = (compass.getValues()[0]*90)*(math.pi/180)
    if not math.isnan(initial_heading):
        break

# Define the waypoints as a list of (x, y) tuples for goal objects
# waypoints_map = [(abs(int(x*12)), abs(int(y*12))) for x, y in waypoints]
# waypoints converted into map coordinates already
# waypoints_map = [(134, 18),  (219, 65), (212, 85), (165, 85),
#                  (140, 65), (140, 130), (110, 130), (207, 161)]
waypoints_map = [(134, 18),  (219, 65), (212, 85),
                 (140, 65), (140, 130), (110, 130), (207, 161)]

waypoint_i = 0  # iterator for waypoints
waypoint_next = False   # iterator for waypoints
waypoint_flag = 0

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
        if in_range:
            return True
    return False


def do_color_filtering(img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    # Create a matrix of dimensions [height, width] using numpy
    # Index mask as [height, width] (e.g.,: mask[y,x])
    mask = np.zeros([img_height, img_width])
    for i in range(img_height):
        for j in range(img_width):
            # print(img[i,j])
            if check_if_color_in_range(img[i, j]) is True:
                mask[i, j] = 1
    return mask


def expand(img_mask, cur_coordinate, coordinates_in_blob):
    if cur_coordinate[0] < 0 or cur_coordinate[1] < 0 or cur_coordinate[0] >= img_mask.shape[0] or cur_coordinate[1] >= img_mask.shape[1]:
        return
    if img_mask[cur_coordinate[0], cur_coordinate[1]] == 0.0:
        return

    img_mask[cur_coordinate[0], cur_coordinate[1]] = 0
    coordinates_in_blob.append(cur_coordinate)

    above = [cur_coordinate[0]-1, cur_coordinate[1]]
    below = [cur_coordinate[0]+1, cur_coordinate[1]]
    left = [cur_coordinate[0], cur_coordinate[1]-1]
    right = [cur_coordinate[0], cur_coordinate[1]+1]
    for coord in [above, below, left, right]:
        expand(img_mask, coord, coordinates_in_blob)


def expand_nr(img_mask, cur_coord, coordinates_in_blob):
    coordinates_in_blob = []
    # List of all coordinates to try expanding to
    coordinate_list = [cur_coord]
    while len(coordinate_list) > 0:
        # Take the first coordinate in the list and perform 'expand' on it
        cur_coordinate = coordinate_list.pop()
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
    blobs_list = []  # List of all blobs, each element being a list of coordinates belonging to each blob
    for i in range(img_mask_height):
        for j in range(img_mask_width):
            if img_mask[i, j] == 1:
                coord = [i, j]
                blob_coords = []
                blob_expand = expand_nr(mask_copy, coord, blob_coords)
                blobs_list.append(blob_expand)
    return blobs_list


def identify_cube():
    img = camera.getImageArray()  # Read image from current frame of robot camera
    img = np.asarray(img, dtype=np.uint8)
    add_color_range_to_detect([158, 167, 0], [222, 222, 73])  # Detect yellow
    # Create img_mask of all foreground pixels, where foreground is defined as passing the color filter. Same function as homework 3
    img_mask = do_color_filtering(img)
    # Find all the blobs in the img_mask. Same function as homework 3
    blobs = get_blobs(img_mask)
    # Print location of detected object using Webots API
    if len(blobs) > 0:
        obj = camera.getRecognitionObjects()[0]
        position = obj.getPosition()
        print("Cube Location In Relation To Camera:")
        print(position[0], position[1])
    return

#### ========== Smoothing algo  ===== #####


def euclidian(p1, p2):
    # euclidian distance between 2 points
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def smooth_path(path, threshold):
    # Simple Path smoothing algo
    # If the path has only two points
    if len(path) <= 2:
        return path
    # get the outlier point from the path
    max_distance = 0
    index = 0
    for i in range(1, len(path)-1):
        # get the distance from the beggining to the end of the path
        distance = euclidian(path[i], path[0]) + euclidian(path[i], path[-1])
        if distance > max_distance:
            # Save the index of the point
            index = i
            # update the max distance
            max_distance = distance

    # If point greater than threshold split path and smooth
    if max_distance > threshold:
        # left segment of path
        left = path[:index+1]
        # right segment of path
        right = path[index:]
        # Recursively smooth the both sides of array
        smooth_left = smooth_path(left, threshold)
        smooth_right = smooth_path(right, threshold)
        # append and return complete path
        smooth_complete = smooth_left[:-1] + smooth_right
        return smooth_complete
    else:
        # If the point within threshold
        return [path[0], path[-1]]


# ------------------------------------------------------------------
# Planning Algorithm   ---- Planner ----
# Tier 1: Teleoperation: use the keyboard to direct robot to waypoints
# Tier 2: A*
# Tier 3: RRT


class Node:
    # Init a node structure
    def __init__(self, x, y):
        # Init the x and y coordinates for each point
        self.x = x
        self.y = y
        # keep track of the parent of each node
        self.parent = None


class RRT:
    # class init
    def __init__(self, start, end, obstacles, k=1500, delta_q=10):
        self.start = Node(start[0], start[1])  # Start node
        self.end = Node(end[0], end[1])  # end node
        self.obstacles = obstacles  # config_map
        self.k = k  # maximum number of iterations
        self.delta_q = delta_q  # delta_q_q
        self.bound_x = obstacles.shape[0]-1
        self.bound_y = obstacles.shape[1]-1
        # a list of node_list to create a path from
        self.node_list = [self.start]

    def check_valid_vertex(self, x, y):
        # if x < 0 or x >= self.state_bounds[0] or y < 0 or y >= self.state_bounds[1]:
        #     return False
        # elif self.obstacle_map[x][y] == 1:
        #     return False
        # return True
        # if x <= self.bound_x and y <= self.bound_y:
        #     x = self.bound_x
        #     y = self.bound_y
        if self.obstacles[x][y] == 1:  # check if the current point is not an obstacle
            return False
        return True

    def distance(self, n1, n2):
        # lin.alg.norm i.e. distance between 2 points
        # return np.linalg.norm(q_point - node_list[i].point)
        return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

    def get_nearest_vertex(self, curr_node):
        # iterate through the node_list list and calculate the dostance from curr_node
        distances = [self.distance(curr_node, node) for node in self.node_list]
        # get the smallest distance
        nearest_idx = np.argmin(distances)
        # return the node in node_list with the smallest distance
        return self.node_list[nearest_idx]

    def get_random_valid_vertex(self):
        # vertex = None
        # while vertex is None:  # Get starting vertex
        # generate a random x coordinate bounded by the shape
        x = random.randint(0, self.bound_x)
        # generate a random y coord. bounded by the map shape
        y = random.randint(0, self.bound_y)
        #     if state_is_valid((x, y)):  # Check if vertex is valid
        #         vertex = (x, y)
        # return vertex
        # crate a random node with given coord.
        new_node = Node(x, y)
        # get the nearest node in node_list from the new_node
        nearest_node = self.get_nearest_vertex(new_node)
        # get the distance between the nodes
        dist = self.distance(new_node, nearest_node)
        # if the dist is bigger than our threshold update
        if dist > self.delta_q:
            # get the distance for the x coord
            x = int(nearest_node.x + (new_node.x - nearest_node.x)
                    * self.delta_q / dist)
            new_node.x = x
            # get the distance for the y coord
            y = int(nearest_node.y + (new_node.y - nearest_node.y)
                    * self.delta_q / dist)
            new_node.y = y
        # update the parent node
        new_node.parent = nearest_node
        # return the newly created node
        return new_node

    def plan(self):
        # no need to add Node object again, already done in __init__()
        # iterate through the number of max iterations k
        for i in range(self.k):
            # steer from the nearest node towards the random point
            # new_point = self.steer(nearest_node, rand_point)
            # get a new node
            new_node = self.get_random_valid_vertex()
            # check that the new path is valid
            if self.check_valid_vertex(new_node.x, new_node.y):
                # if node valid, append to the node_list
                self.node_list.append(new_node)
            # check the distance between the new node and the goal/end node
            if self.distance(new_node, self.end) < self.delta_q:
                # if dist < delta_q terminane and return path
                self.end.parent = new_node
                return self.get_path()
        return None

    def get_path(self):
        # init path array
        path = []
        # start from the end node
        current_node = self.end
        # iterathe through the parents of each node in node_list
        while current_node.parent is not None:
            # append current node to path
            path.append((current_node.x, current_node.y))
            # get the curr_node parent as curr_node
            current_node = current_node.parent
        # Append start node with null parent
        path.append((self.start.x, self.start.y))
        # reverse the path before returning
        return path[::-1]


if mode == 'planner':
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
    # ax.imshow(config_space, cmap='binary')
    # Show the plot
    # plt.show()

    # just visit a subset of the end locations for now...
    visit = [(134, 18), (140, 65), (140, 130), (207, 161)]

    start_m = (256, 96)
    end = (134, 18)
    # Create an instance of the RRT class and find a path
    path = []

    while path == []:
        # iterate until you get a valid path
        # Create an instance of the RRT class and find a path
        rrt = RRT(start_m, end, config_space)
        # get the path
        path = rrt.plan()
    # Iterate througha subset up nodes for RRT
    for i in range(len(visit)-1):
        sub_path = []
        while sub_path == []:
            start = (visit[i][0], visit[i][1])
            end = (visit[i+1][0], visit[i+1][1])
            rrt = RRT(start, end, config_space)
            sub_path = rrt.plan()
        path.extend(sub_path[:-1])

    if path:
        # print("Path found:", path)
        threshold = 3.0
        path = smooth_path(path, threshold)
        # Plot the path and the obstacles
        plt.imshow(config_space, cmap='binary')
        # plt.imshow(config_space)
        plt.plot(start_m[1], start_m[0], 'go')
        plt.plot(end[1], end[0], 'ro')

        for waypoint in waypoints_map:
            plt.plot(waypoint[1], waypoint[0], 'ro')

        x, y = zip(*path)
        plt.plot(y, x, '-b')
        plt.show()

        path_w = []
        for node in path:
            x = (abs(int(node[0]/12)))
            y = (abs(int(node[1]/12)))
            # create a node for the world coordinate system
            node_w = (x, y)
            # paint the world path
            ax.scatter(x, y)
            path_w.append(node_w)
        # save the world coordinate waypoint to disk
        # np.save("path.npy", path_w)
    else:
        print("path failed...")
        print(path)
# ------------------------------------------------------------------

gripper_status = "closed"

# Main Loop
while robot.step(timestep) != -1 and mode != 'planner':
    key = keyboard.getKey()
    ###################
    #
    # Mapping
    #
    ###################

    # Ground truth pose
    pose_x = 15 - gps.getValues()[0]  # adjust the origin correctly
    pose_y = 8 - gps.getValues()[1]

    n = compass.getValues()
    rad = -((math.atan2(n[1], n[0]))-1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()  # get all lidar scans
    lidar_sensor_readings = lidar_sensor_readings[83:len(
        lidar_sensor_readings)-83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]  # add correct offset

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi

        # Convert detection from robot coordinates into world coordinates
        wx = math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy = math.sin(t)*rx + math.cos(t)*ry + pose_y

        ################ ^ [End] Do not modify ^ ##################

        if wx >= 30:
            wx = 29.999
        if wy >= 16:
            wy = 15.999
        if rho < LIDAR_SENSOR_MAX_RANGE:

            # ---- Part 1.3: visualize map gray values. ----
            x = abs(int(wx*12))
            y = abs(int(wy*12))

            increment_value = 5e-3
            map[x, y] += increment_value

            # make sure the value does not exceed 1
            g = min(map[x][y], 1.0)

            # convert the gray scale
            # set g to a vallue depending on our map
            color = (g*256**2+g*256+g)*255

            # draw map on diplsay
            display.setColor(int(color))
            display.drawPixel(y, x)

    # draw the robots line
    display.setColor(int(0xFF0000))
    display.drawPixel(abs(int(pose_y*12)), abs(int(pose_x*12)))

    if mode == 'manual':  # manual mode using arrow keys to drive robot
        while (keyboard.getKey() != -1):
            pass
        if key == keyboard.LEFT:
            vL = -MAX_SPEED  # rotate left
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED  # rotate right
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED  # go forward
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED  # go back
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0  # stop the robot
            vR = 0
        elif key == ord('S'):  # save the map

            threshold_value = 0.5
            thresholded_map = np.multiply(map > threshold_value, 1)
            map_name = 'map.npy'

            # save the thresholded map data to a file
            map_trimmed = thresholded_map[0:800, 0:360]
            np.save(map_name, map_trimmed)
            print("Map file saved as %s" % (map_name))

        elif key == ord('L'):  # load the map
            map = np.load("map.npy")
            plt.imshow(np.fliplr(map))
            plt.title('Map')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
            print("Map loaded...")
        elif key == ord('P'):
            # switch to manual mode
            print("-> Swithing to Autonomous path following")
            mode = 'path_following'
            # path_follow = True
        # CV vision
        elif key == ord('C'):
            identify_cube()
        else:  # slow down
            vL *= 0.75
            vR *= 0.75

    if mode == 'autonomous':
        # want the robot to rotate 90deg everytime it gets too close to a wall infront of it
        desired_rotation = 1.57

        if (beginning == 0):  # make sure the robot initially begins at a standstill
            vL = 0
            vR = 0
            beginning = 1

        # incrementing the counter by 0.5 so everytime it has a remainder of 0.5 it won't enter this if
        elif (wall_counter % 1 == 0):
            # convert the compass angle to an understandable degree measurment
            current_heading = (compass.getValues()[0]*90)*(math.pi/180)

            # these wall_counters are for all the slight edge cases I had to account for
            # since the compass values aren't great, I had to account for many different edge cases depending on which section of the mapping it was at
            if (wall_counter == 0 or wall_counter == 1):
                rotation = current_heading - initial_heading
            elif (wall_counter == 2 or wall_counter == 4 or wall_counter == 8):
                rotation = current_heading
            elif (wall_counter == 3 or wall_counter == 7):
                rotation = current_heading - initial_heading + 0.096
            elif (wall_counter == 5 or wall_counter == 9):
                rotation = current_heading - initial_heading + 0.045
            elif (wall_counter == 6):
                rotation = current_heading
            elif (wall_counter == 7):
                rotation = current_heading - (initial_heading + 0.0162)

            # rotate on the spot
            vL = 0.2*MAX_SPEED
            vR = -0.2*MAX_SPEED
            # stop rotating once the rotation is greater than 90 degrees
            if abs(rotation) >= desired_rotation:
                wall_counter += 0.5
                # rotate on the spot to offset the steering error that gets created when trying to go straight again
                robot_parts["wheel_left_joint"].setVelocity(-0.2*MAX_SPEED)
                robot_parts["wheel_right_joint"].setVelocity(0.2*MAX_SPEED)
                robot.step(864)
                # stop the robot
                robot_parts["wheel_left_joint"].setVelocity(0)
                robot_parts["wheel_right_joint"].setVelocity(0)
                robot.step(1000)

                # collect the current x and y coordinates of the robot
                initial_x = gps.getValues()[0]
                initial_y = gps.getValues()[1]

        else:
            # while the robot isn't too close to the wall infront of it
            if (lidar_sensor_readings[250] > 2.1):
                # keep going straight
                vL = MAX_SPEED
                vR = MAX_SPEED
                # there is a stage counter that tracks which stage of the mapping the robot is currently at
                if (stage_counter == 5 or stage_counter == 9):
                    current_x = gps.getValues()[0]
                    # calculate the x difference to see how far it has traveled
                    delta_x = math.copysign(
                        1, current_x) * abs(current_x - initial_x)
                    if (abs(delta_x) >= 18.3):  # if it travels more than 18.3m in the x-direction
                        # increment counters
                        wall_counter += 0.5
                        stage_counter += 1
                        initial_heading = (compass.getValues()[
                                           0]*90)*(math.pi/180)
                        if (stage_counter == 10):  # finished mapping if at this stage
                            threshold_value = 0.5
                            thresholded_map = np.multiply(
                                map > threshold_value, 1)
                            map_name = 'map.npy'  # save the map to map.npy

                            # save the thresholded map data to a file
                            map_trimmed = thresholded_map[0:800, 0:360]
                            np.save(map_name, map_trimmed)
                            print("Map file saved as %s" % (map_name))
                            robot_parts["wheel_left_joint"].setVelocity(
                                0)  # stop the robot
                            robot_parts["wheel_right_joint"].setVelocity(0)
                            break

                # this elif is for getting from one corridor to the next in the middle
                elif (stage_counter == 6):
                    if (lidar_sensor_readings[250] <= 5.45):
                        wall_counter += 0.5
                        stage_counter += 1
                        initial_heading = (compass.getValues()[
                                           0]*90)*(math.pi/180)

                # this elif is for getting from the other middle corridor to the next
                elif (stage_counter == 8):
                    current_y = gps.getValues()[1]
                    delta_y = math.copysign(
                        1, current_y) * abs(current_y - initial_y)
                    if (abs(delta_y) >= 2.3):  # traveled more than 2.3m in the y direction
                        wall_counter += 0.5
                        stage_counter += 1
                        initial_heading = (compass.getValues()[
                                           0]*90)*(math.pi/180)
            else:
                wall_counter += 0.5

                # pause the robot
                robot_parts["wheel_left_joint"].setVelocity(0)
                robot_parts["wheel_right_joint"].setVelocity(0)
                robot.step(1000)

                initial_heading = (compass.getValues()[0]*90)*(math.pi/180)

                stage_counter += 1
    # move the robot at 0.5 the max speed
    robot_parts["wheel_left_joint"].setVelocity(0.5*vL)
    robot_parts["wheel_right_joint"].setVelocity(0.5*vR)
    ### =========== =========== =========== ###
    ### =========== Path Following =========== ###
    ### =========== =========== =========== ###
    while robot.step(timestep) != -1 and mode == 'path_following':
        pose_y = 15 - gps.getValues()[0]
        pose_x = 8 - gps.getValues()[1]

        n = compass.getValues()
        rad = -((math.atan2(n[1], n[0]))-1.5708)
        pose_theta = rad

        # Create an instance of the RRT class and find a path
        path = []
        path_w = []
        start_n = abs(int(pose_y*12)), abs(int(pose_x*12))
        waypoint_n = (0, 0)

        if waypoint_i == 0 and waypoint_flag == 0:
            print("-> Getting first waypoint")
            # waypoint_flag = 1   # get the path for first waypoint
            # start_n = (256, 96)
            waypoint_n = (134, 18)
        # print("-> Getting New wypoints:")
        else:
            # move on to the next waypoint
            if waypoint_next == True:
                print("-> Init Traversing to next waypoint")
                waypoint_i += 1
                waypoint_next = False
                # get the map for next waypoint
                waypoint_flag = 0
            # print("start_n: ", start_n[0], start_n[1])
            # print("waypoint_n: ", waypoint_n[0], waypoint_n[1])
            if waypoint_i == len(waypoints_map):
                # stop iteration
                print("-> Finished iteration")
                path_follow = False
                waypoint_i = 0
                mode = "manual"

            # load curr waypoint
            # start_n = (waypoints_map[waypoint_i][0],
            #    waypoints_map[waypoint_i][1])
            # get the curr iteration of waypoints
            waypoint_n = (waypoints_map[waypoint_i][0],
                          waypoints_map[waypoint_i][1])
        # Load the first waypoint only
        key = keyboard.getKey()
        while (keyboard.getKey() != -1):
            pass
        # elif key == ord('N'):
        if key == ord('R'):
            print("-> Swithing to Autonomous path following")
            path_follow = True
        elif key == ord('M'):
            # switch to manual mode
            print("Swithing to manual mode")
            mode = 'manual'
            path_follow = False

        # and start_n != (0, 0) and waypoint_n != (0, 0)
        if path_follow == False and waypoint_flag == 0:
            print("-> Getting path to waypoint")
            # load the map from disk
            map = np.load('./config_space.npy')
            # waypoint flag to false
            waypoint_flag = 1
            # init arbitrary value
            min_path = [(-1, -1)] * 50
            # get the smoothest path possible i.e. the shortest
            for i in range(100):
                while path is None or path == []:
                    rrt = RRT(start_n, waypoint_n, map)
                    path = rrt.plan()
                # get the minimum path only
                if len(path) <= len(min_path):
                    min_path = path

            print("getting valid path")
            if path and min_path[0] != (-1, -1):
                path = min_path
                # Apply smoothin algorithm to path
                threshold = 7.0
                path = smooth_path(path, threshold)
                # apply smoothong algo with a bigger threshold
                threshold = 10.0
                path2 = smooth_path(path, threshold)

                # Create Figure
                fig, ax = plt.subplots()
                # Plot the configuration space
                ax.imshow(map, cmap='binary')
                # Plot the path and the obstacles
                plt.plot(start_n[1], start_n[0], 'go')
                plt.plot(waypoint_n[1], waypoint_n[0], 'ro')
                # plt.imshow(config_space)
                x, y = zip(*path2)
                plt.plot(y, x, '-b')
                plt.show()
                # convert the coordinates back to world coordinates
                for i in range(len(path2)-1):
                    x = (path2[i][0]/12)
                    y = (path2[i][1]/12)
                    # create a node for the world coordinate system
                    node_w = (x, y)
                    # paint the world path
                    # ax.scatter(x, y)
                    path_w.append(node_w)
                # save the world coordinate waypoint to disk
                np.save("single_path.npy", path_w)
                # path_follow = True
            else:
                print("path failed...")
                print(path)
        # Implement feedback control loop
        if path_follow:
            # ODOMETRTY Code taken from piazza
            distL = vL/MAX_SPEED * MAX_SPEED_MS * timestep/1000.0
            distR = vR/MAX_SPEED * MAX_SPEED_MS * timestep/1000.0

            pose_x -= (distL+distR) / 2.0 * math.cos(pose_theta)
            pose_y -= (distL+distR) / 2.0 * math.sin(pose_theta)
            pose_theta += (distR-distL)/AXLE_LENGTH

            # Bound pose_theta between [-pi, 2pi+pi/2]
            # Important to not allow big fluctuations between timesteps (e.g., going from -pi to pi)
            if pose_theta > 6.28+3.14/2:
                pose_theta -= 6.28
            if pose_theta < -3.14:
                pose_theta += 6.28

            # load the most recent single_path
            path = np.load('single_path.npy').tolist()
            # switch the coordinates to match world
            for i in range(len(path)):
                path[i] = (path[i][1], path[i][0])

            # get the next tuple in path
            goal_x = path[path_i][0]
            goal_y = path[path_i][1]

            # # Calculate the error between the position and the path waypoint
            # # STEP 1: Calculate the error
            rho = math.sqrt((pose_x - goal_x)**2 + (pose_y - goal_y)**2)
            alpha = (math.atan2((goal_x-pose_x), (goal_y-pose_y))-pose_theta)
            goal_theta = alpha
            ## Heading Error: nu ##
            nu = goal_theta - pose_theta  # angle to point in correct direction

            # Clamp error values
            if alpha < -3.1415:
                alpha += 6.283
            if nu < -3.1415:
                nu += 6.283
            # feedback control

            # # Completly dictate on alpha
            p1, p2, p3 = 0, 0, 0
            if (abs(alpha) > 5):
                p1 = 0  # small gains for rho
                p2 = 10  # higher gains for alpha
                p3 = 0  # no update for nu
            else:
                # Prioratize position error
                if (abs(rho) > 1):
                    p1 = 5  # higher gains for rho i.e. position
                    p2 = 5
                    p3 = 0
                else:
                    # Prioritize heading error
                    p1 = 1
                    p2 = 1
                    p3 = 10
            # # Controller
            # to calculate velocities
            xR_dot = 0
            thetaR_dot = 0
            radius = (MAX_SPEED_MS/MAX_SPEED)
            d = AXLE_LENGTH
            # update values depending on gains
            xR_dot = p1 * rho
            thetaR_dot = p2 * alpha + p3 * nu
            # STEP 3: Compute wheelspeeds
            vL = xR_dot + ((thetaR_dot/2)*d)  # Left wheel velocity in rad/s
            vR = xR_dot - ((thetaR_dot/2)*d)

            # # sopping criteria
            # STEP 2.8 Create stopping criteria
            if abs(rho) <= 1:
                path_i += 1
                if path_i == len(path)-1:
                    # stop to move on to the next waypoint
                    vL = 0
                    vR = 0
                    print("-> Switching to manipulation")
                    # update flags
                    mode = 'manual'
                    waypoint_next = True
                    path_follow = False
                    path_i = 0
                # print("Curr:", path[path_i][0], path[path_i][1])
                # print("Next:", path[path_i+1][0], path[path_i+1][1])

            # Actuator commands
            robot_parts["wheel_left_joint"].setVelocity(0.2*vL)
            robot_parts["wheel_right_joint"].setVelocity(0.2*vR)

            # print(f'Waiptoint: {path_i=}')
            # print(f'Pose: {pose_x=} {pose_y=} {pose_theta=}')
            # print(f'Goal: {goal_x=} {goal_y=}')
            # print(f'Error values: {rho=} {alpha=}')
            # print(f'Speeds: {vR=} {vL=}')
            # print("---------------------------------")

    if arm_mode == 'manual':  # manual arm mode means that we can adjust the position of all 7 joints in the robotic arm
        while (keyboard.getKey() != -1):
            pass

        # for all of these next elifs, the intent is the same, each number joint is mapped to it's respective number key on the keyboard to move the joint in a positive direction
        # the same is done for moving the joint in a negative direction where the key is mapped to the letter directly below its number
        # so in this case, key 1 moves arm_1_joint up and key Q moves arm_1_joint down. This applies to keys 1-7 and Q-U
        if key == ord('1'):  # if number 1 key is pressed
            # convert the position number to a type list (only way I could work around the error I was getting)
            part_positions_list = list(part_positions)
            # increment the position of the joint by +0.025
            part_positions_list[3] += 0.025
            # if the joint has reached it's max position, cap the value at 2.68
            part_positions_list[3] = min(part_positions_list[3], 2.68)
            # convert the list element back to a tuple for .setPosition()
            part_positions = tuple(part_positions_list)

            # set its new position with the slight increment
            robot_parts["arm_1_joint"].setPosition(float(part_positions[3]))
            robot_parts["arm_1_joint"].setVelocity(
                robot_parts["arm_1_joint"].getMaxVelocity() / 2.0)  # move the joint at this velocity

        elif key == ord('Q'):  # if Q key is pressed
            # convert the position number to a type list (only way I could work around the error I was getting)
            part_positions_list = list(part_positions)
            # decrement the position of the joint by +0.025
            part_positions_list[3] -= 0.025
            # if the joint has reached it's max position, cap the value at 0.07
            part_positions_list[3] = max(part_positions_list[3], 0.07)
            # convert the list element back to a tuple for .setPosition()
            part_positions = tuple(part_positions_list)

            # set its new position with the slight decrement
            robot_parts["arm_1_joint"].setPosition(float(part_positions[3]))
            robot_parts["arm_1_joint"].setVelocity(
                robot_parts["arm_1_joint"].getMaxVelocity() / 2.0)  # move the joint at this velocity

        # the remaining elifs operate in the exact same fashion as the first if else with every couple of elifs being mapped to the same joint, either increment or decrementing the position of the joint
        elif key == ord('2'):
            part_positions_list = list(part_positions)
            part_positions_list[4] += 0.025
            part_positions_list[4] = min(part_positions_list[4], 1.02)
            part_positions = tuple(part_positions_list)

            robot_parts["arm_2_joint"].setPosition(float(part_positions[4]))
            robot_parts["arm_2_joint"].setVelocity(
                robot_parts["arm_2_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('W'):
            part_positions_list = list(part_positions)
            part_positions_list[4] -= 0.025
            part_positions_list[4] = max(part_positions_list[4], -1.5)
            part_positions = tuple(part_positions_list)

            robot_parts["arm_2_joint"].setPosition(float(part_positions[4]))
            robot_parts["arm_2_joint"].setVelocity(
                robot_parts["arm_2_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('3'):
            part_positions_list = list(part_positions)
            part_positions_list[5] += 0.025
            part_positions_list[5] = min(part_positions_list[5], 1.5)
            part_positions = tuple(part_positions_list)

            robot_parts["arm_3_joint"].setPosition(float(part_positions[5]))
            robot_parts["arm_3_joint"].setVelocity(
                robot_parts["arm_3_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('E'):
            part_positions_list = list(part_positions)
            part_positions_list[5] -= 0.025
            part_positions_list[5] = max(part_positions_list[5], -3.46)
            part_positions = tuple(part_positions_list)

            robot_parts["arm_3_joint"].setPosition(float(part_positions[5]))
            robot_parts["arm_3_joint"].setVelocity(
                robot_parts["arm_3_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('4'):
            part_positions_list = list(part_positions)
            part_positions_list[6] += 0.025
            part_positions_list[6] = min(part_positions_list[6], 2.29)
            part_positions = tuple(part_positions_list)

            robot_parts["arm_4_joint"].setPosition(float(part_positions[6]))
            robot_parts["arm_4_joint"].setVelocity(
                robot_parts["arm_4_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('R'):
            part_positions_list = list(part_positions)
            part_positions_list[6] -= 0.025
            part_positions_list[6] = max(part_positions_list[6], -0.32)
            part_positions = tuple(part_positions_list)

            robot_parts["arm_4_joint"].setPosition(float(part_positions[6]))
            robot_parts["arm_4_joint"].setVelocity(
                robot_parts["arm_4_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('5'):
            part_positions_list = list(part_positions)
            part_positions_list[7] += 0.025
            part_positions_list[7] = min(part_positions_list[7], 2.07)
            part_positions = tuple(part_positions_list)

            robot_parts["arm_5_joint"].setPosition(float(part_positions[7]))
            robot_parts["arm_5_joint"].setVelocity(
                robot_parts["arm_5_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('T'):
            part_positions_list = list(part_positions)
            part_positions_list[7] -= 0.025
            part_positions_list[7] = max(part_positions_list[7], -2.07)
            part_positions = tuple(part_positions_list)

            robot_parts["arm_5_joint"].setPosition(float(part_positions[7]))
            robot_parts["arm_5_joint"].setVelocity(
                robot_parts["arm_5_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('6'):
            part_positions_list = list(part_positions)
            part_positions_list[8] += 0.025
            part_positions_list[8] = min(part_positions_list[8], 1.39)
            part_positions = tuple(part_positions_list)

            robot_parts["arm_6_joint"].setPosition(float(part_positions[8]))
            robot_parts["arm_6_joint"].setVelocity(
                robot_parts["arm_6_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('Y'):
            part_positions_list = list(part_positions)
            part_positions_list[8] -= 0.025
            part_positions_list[8] = max(part_positions_list[8], -1.39)
            part_positions = tuple(part_positions_list)

            robot_parts["arm_6_joint"].setPosition(float(part_positions[8]))
            robot_parts["arm_6_joint"].setVelocity(
                robot_parts["arm_6_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('7'):
            part_positions_list = list(part_positions)
            part_positions_list[9] += 0.025
            part_positions_list[9] = min(part_positions_list[9], 2.07)
            part_positions = tuple(part_positions_list)

            robot_parts["arm_7_joint"].setPosition(float(part_positions[9]))
            robot_parts["arm_7_joint"].setVelocity(
                robot_parts["arm_7_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('U'):
            part_positions_list = list(part_positions)
            part_positions_list[9] -= 0.025
            part_positions_list[9] = max(part_positions_list[9], -2.07)
            part_positions = tuple(part_positions_list)

            robot_parts["arm_7_joint"].setPosition(float(part_positions[9]))
            robot_parts["arm_7_joint"].setVelocity(
                robot_parts["arm_7_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('J'):
            part_positions_list = list(part_positions)
            part_positions_list[12] += 0.005
            part_positions_list[13] += 0.005
            part_positions_list[12] = min(part_positions_list[12], 0.045)
            part_positions_list[13] = min(part_positions_list[13], 0.045)
            part_positions = tuple(part_positions_list)

            robot_parts["gripper_left_finger_joint"].setPosition(
                float(part_positions[12]))
            robot_parts["gripper_right_finger_joint"].setPosition(
                float(part_positions[13]))
            robot_parts["gripper_left_finger_joint"].setVelocity(
                robot_parts["gripper_left_finger_joint"].getMaxVelocity() / 2.0)
            robot_parts["gripper_right_finger_joint"].setVelocity(
                robot_parts["gripper_right_finger_joint"].getMaxVelocity() / 2.0)
        # same as above
        elif key == ord('K'):
            part_positions_list = list(part_positions)
            part_positions_list[12] -= 0.005
            part_positions_list[13] -= 0.005
            part_positions_list[12] = max(part_positions_list[12], 0)
            part_positions_list[13] = max(part_positions_list[13], 0)
            part_positions = tuple(part_positions_list)

            robot_parts["gripper_left_finger_joint"].setPosition(
                float(part_positions[12]))
            robot_parts["gripper_right_finger_joint"].setPosition(
                float(part_positions[13]))
            robot_parts["gripper_left_finger_joint"].setVelocity(
                robot_parts["gripper_left_finger_joint"].getMaxVelocity() / 2.0)
            robot_parts["gripper_right_finger_joint"].setVelocity(
                robot_parts["gripper_right_finger_joint"].getMaxVelocity() / 2.0)

        # print('Arm positions: %s, Gripper positions: %s' % (part_positions[3:10], part_positions[12:14]))

    if arm_positioning == 'on':  # checking that we have access to the hot keys that control the position of all joints in one go
        while (keyboard.getKey() != -1):
            pass
        # the following four if statements all behave the exact same ways
        # each if statement moves the robotic arm into a desired and pre-mapped position depending on which task we are trying to accomplish
        # such as grabbing the top shelf object, middle shelf object, dropping in basket or having the arm stowed away
        # since we only want to manipualate the position of the 7 arm joints, we only iterate through the 3 and 9th element in the array and the last two for the gripper position
        if key == ord('G'):  # stowed away position
            part_positions = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32,
                              0.0, 1.41, 'inf', 'inf', 0.045, 0.045)  # desired joint positions
            # loop through all joints
            for i, part_name in enumerate(part_names):
                # check if i is within the range [3, 9] or [12,13] as we only want to manipulate the position of these joints
                if (3 <= i <= 9) or (12 <= i <= 13):
                    robot_parts[part_name].setPosition(
                        float(part_positions[i]))  # set the position of the joint
                    robot_parts[part_name].setVelocity(
                        robot_parts[part_name].getMaxVelocity() / 2.0)  # set position at this velocity

        # the three remaing if statements all behave the exact same way as the first if statement, just with different desired part_position locations
        if key == ord('H'):  # grabbing top shelf object position
            part_positions = (0.0, 0.0, 0.35, 1.595, 0.72, -3.185, -
                              0.32, 1.645, 1.39, 0.12, 'inf', 'inf', 0.045, 0.045)
            for i, part_name in enumerate(part_names):
                # check if i is within the range [3, 9] or [12,13]
                if (3 <= i <= 9) or (12 <= i <= 13):
                    robot_parts[part_name].setPosition(
                        float(part_positions[i]))
                    robot_parts[part_name].setVelocity(
                        robot_parts[part_name].getMaxVelocity() / 2.0)

        if key == ord('J'):  # grabbing middle shelf object position
            part_positions = (0.0, 0.0, 0.35, 1.62, -1.23, -3.185,
                              1.03, 0.145, 0.0, 1.41, 'inf', 'inf', 0.045, 0.045)
            for i, part_name in enumerate(part_names):
                # check if i is within the range [3, 9] or [12,13]
                if (3 <= i <= 9) or (12 <= i <= 13):
                    robot_parts[part_name].setPosition(
                        float(part_positions[i]))
                    robot_parts[part_name].setVelocity(
                        robot_parts[part_name].getMaxVelocity() / 2.0)

        if key == ord('K'):  # dropping object in basket position
            part_positions = (0.0, 0.0, 0.35, 0.495, -0.155, -0.335,
                              2.29, 0.145, 0.0, 1.41, 'inf', 'inf', 0.0, 0.0)
            for i, part_name in enumerate(part_names):
                if (3 <= i <= 9):  # check if i is within the range [3, 9]
                    robot_parts[part_name].setPosition(
                        float(part_positions[i]))
                    robot_parts[part_name].setVelocity(
                        robot_parts[part_name].getMaxVelocity() / 5.0)

while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass
