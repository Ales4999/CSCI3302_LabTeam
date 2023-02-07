"""epuck_avoid_collision controller."""

# You may need to import some classes of the controller module. Ex:
from controller import Robot, DistanceSensor, Motor

##Define the duration of the physics step
TIME_STEP = 64
MAX_SPEED = 6.28


# create the Robot instance.
robot = Robot()

# initialize devices
#get all the sensor names and store in array 
ps = []
psNames = [
    'ps0', 'ps1', 'ps2', 'ps3',
    'ps4', 'ps5', 'ps6', 'ps7'
]
#iterate through to initialize each sensor to TIME_STEP
for i in range(8):
    ps.append(robot.getDevice(psNames[i]))
    ps[i].enable(TIME_STEP)

#Initialize the motors 
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')

#set the position of the motors 
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

#set the velocity of the motors 
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# feedback loop: step simulation until receiving an exit event
while robot.step(TIME_STEP) != -1:
    # read sensors outputs
    psValues = []
    for i in range(8):
        psValues.append(ps[i].getValue())
        
    # process behavior -> # detect obstacles
    #thresshold is 0.05 m away frorm an object
    right_obstacle = psValues[0] > 80.0 or psValues[1] > 80.0 or psValues[2] > 80.0
    left_obstacle = psValues[5] > 80.0 or psValues[6] > 80.0 or psValues[7] > 80.0
    
    # write actuators inputs
    
    # initialize motor speeds at 50% of MAX_SPEED.
    leftSpeed  = 0.5 * MAX_SPEED
    rightSpeed = 0.5 * MAX_SPEED
    # modify speeds according to obstacles
    if left_obstacle:
        # turn right
        leftSpeed  = 0.5 * MAX_SPEED
        rightSpeed = -0.5 * MAX_SPEED
    elif right_obstacle:
        # turn left
        leftSpeed  = -0.5 * MAX_SPEED
        rightSpeed = 0.5 * MAX_SPEED
    # write actuators inputs
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    
    # exit success
    pass

# Enter here exit cleanup code.
