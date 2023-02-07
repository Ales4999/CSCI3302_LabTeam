"""Lab1_EPuck_Controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, DistanceSensor, Motor
TIME_STEP = 64
MAX_SPEED = 6.28
# create the Robot instance.
robot = Robot()

ps = []
psNames = [
    'ps0', 'ps1', 'ps2', 'ps3',
    'ps4', 'ps5', 'ps6', 'ps7'
]

for i in range(8):
    ps.append(robot.getDevice(psNames[i]))
    ps[i].enable(TIME_STEP)

leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    psValues = []
    for i in range(8):
        psValues.append(ps[i].getValue())
    # Process sensor data here.
    headOn_Obstacle = psValues[0] < 50 or psValues[7] < 50
    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    leftSpeed = 4
    rightSpeed = 4
    if headOn_Obstacle:
        leftSpeed = 1
        rightSpeed = -1
        
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)

# Enter here exit cleanup code.
