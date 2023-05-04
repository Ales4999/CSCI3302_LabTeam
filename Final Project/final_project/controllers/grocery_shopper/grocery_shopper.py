"""grocery controller."""

# Nov 2, 2022
# Group: WALL-E's
# Team Members:
# Mark Abbott
# Miles Sanders
# Alberto Espinosa

from controller import Robot, Keyboard
import math
import random
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

# mode = 'manual'
mode = 'planner'
# mode = 'autonomous'

map = np.zeros(shape=[360, 192])
# map = np.zeros(shape=[372, 372])

# waypoints = [(11.20, 1.55), (18.26, 5.05),
#              (17.74, 7.00), (13.97, 7.24), (12.51, 5.38), (11.28, 5.29),
#              (9.34, 10.73), (12.34, 10.73), (17.28, 13.50)]


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

    def check_node_valid(self, x, y):
        if self.obstacles[x][y] == 1:  # check if the current point is not an obstacle
            return False
        return True

    def distance(self, n1, n2):
        # lin.alg.norm i.e. distance between 2 points
        return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

    def nearest_node(self, curr_node):
        # iterate through the node_list list and calculate the dostance from curr_node
        distances = [self.distance(curr_node, node) for node in self.node_list]
        # get the smallest distance
        nearest_idx = np.argmin(distances)
        # return the node in node_list with the smallest distance
        return self.node_list[nearest_idx]

    def new_node(self):
        # generate a random x coordinate bounded by the shape
        x = random.randint(0, self.bound_x)
        # generate a random y coord. bounded by the map shape
        y = random.randint(0, self.bound_y)
        # crate a random node with given coord.
        new_node = Node(x, y)
        # get the nearest node in node_list from the new_node
        nearest_node = self.nearest_node(new_node)
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

    def find_path(self):
        # iterate through the number of max iterations k
        for i in range(self.k):
            # get a new node
            new_node = self.new_node()
            # check that the new path is valid
            if self.check_node_valid(new_node.x, new_node.y):
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
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    # start_x = 19.79
    start_x = 21.4
    # start_y = 8
    start_y = 8
    start_w = (start_x,  start_y)  # (Pose_X, Pose_Y) in meters

    # End coordinates (134, 18)
    end_x = 11.1666667
    end_y = 1.5
    end_w = (end_x, end_y)  # (Pose_X, Pose_Y) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    # (x, y) in 360x360 map
    start = (abs(int(start_w[0]*12)), abs(int(start_w[1]*12)))
    # print("Start node: ", start)
    # (x, y) in 360x360 map
    end = (abs(int(end_w[0]*12)), abs(int(end_w[1]*12)))
    # print("End node: ", end)

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

    # Define the waypoints as a list of (x, y) tuples
    # Convert tuples from world coordinate into map coordinates
    # waypoints_map = [(abs(int(x*12)), abs(int(y*12))) for x, y in waypoints]

    # waypoints converted into map coordinates already
    waypoints_map = [(134, 18),  (219, 65), (212, 85),
                     (165, 85), (140, 65), (140, 130), (110, 130), (207, 161)]
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
        path = rrt.find_path()

    for i in range(len(visit)-1):
        sub_path = []
        while sub_path == []:
            start = (visit[i][0], visit[i][1])
            end = (visit[i+1][0], visit[i+1][1])
            rrt = RRT(start, end, config_space)
            sub_path = rrt.find_path()
        path.extend(sub_path[:-1])

    if path:
        # print("Path found:", path)

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
# Helper Functions


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
    lidar_sensor_readings = lidar_sensor_readings[83:len(
        lidar_sensor_readings)-83]

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
        # print(wx, wy)

        ################ ^ [End] Do not modify ^ ##################

        # print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f, x: %f, y: %f" % (rho,alpha,rx,ry,wx,wy,x,y))
        # print(wx,wy)
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
            # print(x,y)
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
    display.drawPixel(abs(int(pose_y*12)), abs(int(pose_x*12)))

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

    # Actuator commands
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

    print("X: %f Y: %f Theta: %f" % (pose_x, pose_y, pose_theta))


while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass
