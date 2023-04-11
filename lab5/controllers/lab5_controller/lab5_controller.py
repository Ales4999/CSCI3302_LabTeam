"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
# Uncomment if you want to use something else for finding the configuration space
from scipy.signal import convolve2d
import heapq
# from queue import PriorityQueue


MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633  # [m/s]
AXLE_LENGTH = 0.4044  # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75  # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16,
              1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts = []

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

# range = robot.getDevice('range-finder')
# range.enable(timestep)
# camera = robot.getDevice('camera')
# camera.enable(timestep)
# camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

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
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
# mode = 'manual'  # Part 1.1: manual mode
# mode = 'planner'
mode = 'autonomous'


###################
#
# Planner
#
###################
if mode == 'planner':
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    # use the odometry from part 1?
    # wx: -9.711103 wy: -4.925535
    # start_w = (-6.839398,  -6.034829)  # (Pose_X, Pose_Y) in meters
    start_w = (-7.866666,  -4.666667)  # (Pose_X, Pose_Y) in meters

    # corridor outside the bathroom: wx: -6.835060 wy: -7.936140
    end_w = (-7, -10.3,)  # (Pose_X, Pose_Y) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    # (x, y) in 360x360 map
    start = (abs(int(start_w[0]*30)), abs(int(start_w[1]*30)))
    # (x, y) in 360x360 map
    end = (abs(int(end_w[0]*30)), abs(int(end_w[1]*30)))

    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    def path_planner(map, start, end):
        '''
        :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
        :param start: A tuple of indices representing the start cell in the map
        :param end: A tuple of indices representing the end cell in the map
        :return: A list of tuples as a path from the given start to the given end in the given maze
        '''

        # check if the start and end nodes are valid
        # if the start or end nodes are obstacles
        if map[start[0], start[1]] == 1 or map[end[0], end[1]] == 1:
            print("-> Err: start or end nodes are obstacles")
            print("-> Err: start node: %s, and end node: %s " %
                  (map[start[0], start[1]], map[end[0], end[1]]))
            return []
        # if the nodes are out of bounds
        if start[0] < 0 or start[0] >= map.shape[0] or start[1] < 0 or start[1] >= map.shape[1]:
            print("start node is out of bounds")
            return []
        if end[0] < 0 or end[0] >= map.shape[0] or end[1] < 0 or end[1] >= map.shape[1]:
            print("end node is out of bounds")
            return []

        # init dictionaries to keep track of distance from start and predecessor
        distances = {}
        parent = {}

        # Initialize distances
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                # free space
                if map[i, j] == 0:
                    distances[(i, j)] = float('inf')
                    parent[(i, j)] = None
                # obstacle, do not include in map
                else:
                    distances[(i, j)] = -1
                    parent[(i, j)] = -1

        # init the first node
        distances[start] = 0
        parent[start] = None
        # initialize the priority queue with the start node
        pq = []
        heapq.heappush(pq, (0, start))

        # keep track of the visited nodes
        visited = set()

        while pq:
            # Get the node with the smallest distance from the start
            curr_dist, curr_node = heapq.heappop(pq)

            # Check if we have already visited the node
            if curr_node in visited:
                continue

            # mark the curr node as visited
            visited.add(curr_node)

            # Check if we have reached the end node
            if curr_node == end:
                break

            # Update distances to adjacent nodes
            for neighbor in [(curr_node[0]-1, curr_node[1]), (curr_node[0]+1, curr_node[1]), (curr_node[0], curr_node[1]-1), (curr_node[0], curr_node[1]+1)]:
                # check that the neighbor is within the map boundaries
                if neighbor[0] < 0 or neighbor[0] >= map.shape[0] or neighbor[1] < 0 or neighbor[1] >= map.shape[1]:
                    continue
                # check that the neighbor is not an obstacle
                if map[neighbor[0], neighbor[1]] == 1:
                    continue

                # calculate new distance to neighbor
                new_dist = curr_dist + 1
                # if the new distance is shorter, update distance and add to priority queue
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    parent[neighbor] = curr_node
                    # pq.put((new_dist, neighbor))
                    heapq.heappush(pq, (new_dist, neighbor))

        # If we didn't reach the end node, there is no path
        if end not in visited:
            print("End node not found")
            return []

        # Trace back the path from end to start
        dij_path = [end]
        # get a valid path from start to end
        while dij_path[-1] != start:
            # last node
            last_node = dij_path[-1]
            # add its parent to the path
            dij_path.append(parent[last_node])

        # Reverse the path and return it
        dij_path.reverse()
        # print("-> dij_path: ", dij_path)
        # make the path a list of tuples
        path_tuples = [(node[1], node[0]) for node in dij_path]
        # return a list of tuples
        return path_tuples

    # Part 2.1: Load map (map.npy) from disk and visualize it
    map = np.load('./map.npy')

    # visualize the map using matplotlib
    # plt.imshow(np.fliplr(map))
    # plt.title('Map')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()  # comment to not show map

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

    # Part 2.3 continuation: Call path_planner

    # start: 235, 140
    # END: map[309][210]: 0.0
    # for i in range(290, 310):
    #     for j in range(configuration_space.shape[1]):
    #         if configuration_space[i][j] == 0 and j > 100:
    #             print("map[%i][%i]: %s" % (i, j, configuration_space[i][j]))

    # debug with a smaller array
    # test_arr = [[0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 1, 1, 0, 0, 1, 1, 0],
    #             [0, 1, 1, 0, 1, 1, 1, 0],
    #             [0, 1, 1, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 1, 1, 0, 1],
    #             [0, 0, 0, 0, 1, 1, 1, 1],
    #             [0, 0, 0, 0, 1, 1, 1, 1]]

    # start = (0, 2)
    # end = (5, 6)
    # test_arr = np.array(test_arr)

    # end = (225, 181)
    # end = (181, 225)

    # print("-> Start %s and end %s nodes: " % (start, end))

    # path = path_planner(test_arr, start, end)
    path = path_planner(configuration_space, start, end)

    # check if valid
    if (path == []):
        print("-> Invalid path...")
    # else:
    #     print("-> Printing the path from algo: ")
    #     print(path)

    # Part 2.4  convert back to world coordinates
    # (abs(int(start_w[0]*30)), abs(int(start_w[1]*30)))

    # # extract x and y coordinates from the tuples
    # x_coords = [point[0] for point in path]
    # y_coords = [point[1] for point in path]

    # create a scatter plot of the points
    # fig, ax = plt.subplots()
    ax.imshow(config_space, cmap='binary')

    # plot each tuple in the list
    # for d in path:
    #     x, y = d
    #     ax.plot(x, y, 'o')    # show the plot
    # plt.show()

    # np.save("config_space_path", config_space)

    path_w = [(x/30, y/30) for x, y in path]
    np.save("path.npy", path_w)
    print("Path Saved succesfully")


######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
# map = np.zeros(shape=[360, 360])
# notice that the floor is 12m by 12m...
map = np.zeros(shape=[360, 360])
# map = np.zeros(shape=[372, 372])

waypoints = []

if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    # load path from disk
    path = np.load('path.npy')
    visual_map = [(x*30, y*30) for x, y in path]

    # load config file
    config_space = np.load('./config_space.npy')

    # Create a figure
    fig, ax = plt.subplots()
    # Plot the configuration space
    ax.imshow(config_space, cmap='binary')
    # plot each tuple in the list
    for d in visual_map:
        x, y = d
        ax.plot(x, y, 'o')    # show the plot
    # plt.show()
    np.save("config_space_path.npy", config_space)

    # Replace with code to load your path
    waypoints = [(-z, -x) for x, z in path]
    # print(waypoints)


state = 0  # use this to iterate through your path

while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    ################ v [Begin] Do not modify v ##################
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

            # draw robots pose

    # Draw the robot's current pose on the 360x360 display
    # x = 360-abs(int(wx*30))
    # y = abs(int(wy*30))
    # print("-> robot's pose: %f , %f" % (x,y) )

    # draw the robots line
    display.setColor(int(0xFF0000))
    display.drawPixel(abs(int(pose_x*30)), abs(int(pose_y*30)))

    ###################
    #
    # Controller
    #
    ###################
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

            # ---- 1.4 need to clip the map values before drowing them? -----
            # NumPy you can use array>0.5, to get an array of Booleans with “True”
            # entries indicating entries in the original array that satisfy the
            # provided criteria (e.g., all values greater than 0.5).

            # You can then multiply this array (np.multiply) with 1 to convert it back
            # to an integer array. Use NumPy’s save method to store your data structure,
            # making sure to name the file map.npy.

            # set a threshold value
            threshold_value = 0.5

            # print("-> our map values BEFORE clipping:")
            # for row in map:
            # for element in map:
            # print(element, end=" ")
            # print() # Move to the next line after printing each row

            # map = np.clip(map, 0, 1)  # Keep values within [0,1]
            # print("-> our map values AFTER clipping:")
            # print(map)
            # print(map[10][10])

            # threshold the map data to reject all values below threshold_value
            thresholded_map = np.multiply(map > threshold_value, 1)

            # print("-> thresholded_map:")
            # for row in thresholded_map:
            # for element in thresholded_map:
            # print(element, end=" ")
            # print() # Move to the next line after printing each row

            map_name = 'map.npy'

            # save the thresholded map data to a file
            map_trimmed = thresholded_map[0:800, 0:360]
            np.save(map_name, map_trimmed)

            print("Map file saved as %s" % (map_name))

        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
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
    else:  # not manual mode
        # Part 3.2: Feedback controller
        goal_x = waypoints[state][0]
        goal_y = waypoints[state][1]

        # STEP 1: Calculate the error
        rho = math.sqrt((pose_x - goal_x)**2 + (pose_y - goal_y)**2)
        alpha = math.atan2((goal_y-pose_y), (goal_x-pose_x))-pose_theta
        nu = alpha - pose_theta

        # STEP 2: Controller
        # Clamp error values
        if alpha < -3.1415:
            alpha += 6.283
        if nu < -3.1415:
            nu += 6.283

        # need conditional logic to determine controller gains
        # Prioritize Bearing error
        p1, p2, p3 = 0, 0, 0
        if (abs(alpha) > 0.3):
            p1 = 1  # small gains for rho
            p2 = 10  # higher gains for alpha
            p3 = 0  # no update for nu
        else:
            # Prioratize position error
            if (abs(rho) > 0.5):
                p1 = 10  # higher gains for rho i.e. position
                p2 = 1
                p3 = 0
            else:
                # Prioritize heading error
                p1 = 1
                p2 = 1
                p3 = 20

        radius = (MAX_SPEED_MS/MAX_SPEED)
        d = AXLE_LENGTH

        xR_dot = p1 * rho
        thetaR_dot = p2 * alpha + p3 * nu

        # STEP 3: Compute wheelspeeds
        vL = xR_dot - (thetaR_dot/2)*d  # Left wheel velocity in rad/s
        vR = xR_dot + (thetaR_dot/2)*d

        # Normalize wheelspeed
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)

        # 2.7 clamp the velocities if they exceed MAX_SPEED
        if vL > MAX_SPEED:
            vL = MAX_SPEED
        elif abs(vL) > MAX_SPEED:
            vL = - MAX_SPEED

        if vR > MAX_SPEED:
            vR = MAX_SPEED
        elif abs(vR) > MAX_SPEED:
            vR = - MAX_SPEED

        # STEP 2.8 Create stopping criteria
        if abs(rho) <= 0.5 and alpha <= abs(0.5):
            state += 1
            # print("Curr:", waypoints[state][0], waypoints[state][1])
            # print("Next:", waypoints[state+1][0], waypoints[state+1][1])
            if i == 7:
                vL = 0
                vR = 0
                timestep = -1

        # Debugging Code
        print(f'Waiptoint: {state=}')
        print(f'Pose: {pose_x=} {pose_y=} {pose_theta=}')
        print(f'Goal: {goal_x=} {goal_y=} {alpha=}')
        print(f'Error values: {rho=} {alpha=} {nu=}')
        print(f'Speeds: {vR=} {vL=}')
        print("---------------------------------")

    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS * \
        timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS * \
        timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)

while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass
