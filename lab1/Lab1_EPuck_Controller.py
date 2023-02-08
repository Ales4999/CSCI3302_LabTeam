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

##Lab1 pART 2 INST:
# 1. Drive forward
# 2. front sensor reads light intensity of obstacle is aprox 0.05m
# 3. Turn robot 180 degrees
# 4. front sensor reads light intensity of obstacle is aprox 0.05m
# 5. rotate robot clockwise until left distance sensor (ps5) reads <0.05m
# 6. drive forward as long as ps5 reads <0.05m, else stop forever
counter = 0
# feedback loop: step simulation until receiving an exit event
while robot.step(TIME_STEP) != -1:
    # read sensors outputs
    psValues = []
    for i in range(8):
        psValues.append(ps[i].getValue())
        
    # process behavior -> # detect obstacles
    #thresshold is 0.05 m away frorm an object
    front_r = psValues[0] > 80.0 or psValues[1] > 80.0 
    front_l =  psValues[6] > 80.0 or psValues[7] > 80.0
    left_s = psValues[5] > 80.0
    
    #Front obstacle 
    
    # write actuators inputs
    
    # initialize motor speeds at 50% of MAX_SPEED.
    leftSpeed  = 0.5 * MAX_SPEED
    rightSpeed = 0.5 * MAX_SPEED
    # modify speeds according to obstacles
    ## Turn a 180 degrees 
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    
    #Time? or sensors?
    if counter == 0:
    
        if front_r or front_l:
            # turn right
            leftSpeed  = 0.7 * MAX_SPEED
            rightSpeed = -0.7 * MAX_SPEED
            
            #make robot rotate
            leftMotor.setVelocity(leftSpeed)
            rightMotor.setVelocity(rightSpeed)
            
            #set timer for rotation
            #duration = 5000 / timestep 
            
            for i in range(20):
                robot.step(100)
                
            #update the first obstacle
            counter += 1
           
    else:
    
        if left_s:
            #go straight
            leftSpeed  = 0.5 * MAX_SPEED
            rightSpeed = 0.5 * MAX_SPEED
            # modify speeds according to obstacles
            ## Turn a 180 degrees 
            leftMotor.setVelocity(leftSpeed)
            rightMotor.setVelocity(rightSpeed)
            #increase counter to break while loop
            #counter will increase indefenetly
            counter += 1
            #print("increasing counter: ",  counter)
            
        elif front_r or front_l:
            #turn clockwise until ps5 marks <0.05m 
            #only need ps5 sensor 
            # turn right
            leftSpeed  = 0.7 * MAX_SPEED
            rightSpeed = -0.7 * MAX_SPEED
            #print("outside while loop: ", ps[5].getValue() < 80)
                            #make robot rotate
            #print("inside while loop: ", ps[5].getValue() < 80)
            leftMotor.setVelocity(leftSpeed)
            rightMotor.setVelocity(rightSpeed)
            
            #stop rotating when your sensor
            #while left_s:
            #do something
        elif not left_s and counter > 2: 
            #stop forever
            leftMotor.setVelocity(0)
            rightMotor.setVelocity(0)
            break
            # global robot
            # end_time = robot.getTime() + duration
            # while robot.step(TIME_STEP) != 1 and robot.getTime() < end_time
                # pass
            # print("getting inside elif: ",  counter)
         #threshold reached
    
    # exit success
    pass
