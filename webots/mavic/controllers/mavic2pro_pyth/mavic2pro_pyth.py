from controller import Robot
import cnn2
import numpy as np

classes = ['T-Shirts', 'Pants', 'Pullovers', 'Shoes & Sandals', 'Bags']
to_look_for = classes[1]
target_altitude = 0
matches = False

def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)
    
def turn_on_led(led_left, led_right):
    led_left.set(1)
    led_right.set(1)
    print("LED turned on.")
    

def check_if_label_matches(path):
    predicted_label = cnn2.get_label(path)
    print(f'Predicted Class:', predicted_label)
    if predicted_label == to_look_for:
        return True 
    return False
            
def move_to_target(robot, target_points):
        global matches
        if robot.target_position[0:2] == [0, 0]:
            robot.target_position[0:2] = target_points[0] 
              
        target_point = target_points[robot.target_index] 
        x, y = robot.current_pose[:2]
        dist = np.sqrt((target_point[0] - x)**2 + (target_point[1] - y)**2)  
            
        if dist < target_precision and not matches: 
        
            while robot.step(timestep) != -1:
                yaw_disturbance = 1.3
                roll, pitch, yaw = imu.getRollPitchYaw()
                if abs(yaw) > 3.13:
                    break
                    
                        
                roll, pitch, yaw = imu.getRollPitchYaw()
                _, _, altitude = gps.getValues()
                roll_input = K_ROLL_P * clamp(roll, -1, 1) 
                pitch_input = K_PITCH_P * clamp(pitch, -1, 1) 
                yaw_input = yaw_disturbance
                clamped_difference_altitude = clamp(target_altitude - altitude + K_VERTICAL_OFFSET, -1, 1)
                vertical_input = K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

                front_left_motor_input = K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
                front_right_motor_input = K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
                rear_left_motor_input = K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
                rear_right_motor_input = K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

                front_left_motor.setVelocity(front_left_motor_input)
                front_right_motor.setVelocity(-front_right_motor_input)
                rear_left_motor.setVelocity(-rear_left_motor_input)
                rear_right_motor.setVelocity(rear_right_motor_input)

            path = f"image{robot.target_index}.jpg"
            camera.saveImage(path, 100)
            print("Image taken and saved")
            matches = check_if_label_matches(path)
                
            if matches:
                turn_on_led(led_left, led_right)
                robot.target_position[:3] = (robot.target_position[0] + 0.8, robot.target_position[1], 0)    
                robot.target_altitude = 2
            else:
                robot.target_index += 1
                if robot.target_index > len(target_points) - 1:
                    robot.target_index = 0
                robot.target_position[0:2] = target_points[robot.target_index]

             

        robot.target_position[2] = np.arctan2(
            robot.target_position[1] - robot.current_pose[1], robot.target_position[0] - robot.current_pose[0])
        angle_left = robot.target_position[2] - robot.current_pose[5]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if (angle_left > np.pi):
            angle_left -= 2 * np.pi

        yaw_disturbance = MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        pitch_disturbance = clamp(np.log10(abs(angle_left)), MAX_PITCH_DISTURBANCE, 0.1)

        return yaw_disturbance, pitch_disturbance


K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P = 3.0  # P constant of the vertical PID.
K_ROLL_P = 100.0  # P constant of the roll PID.
K_PITCH_P = 30.0  # P constant of the pitch PID.

MAX_YAW_DISTURBANCE = 0.6
MAX_PITCH_DISTURBANCE = -1
target_precision = 0.11  # Precision between the target position and the robot position in meters

robot = Robot()

camera = robot.getDevice("camera")
timestep = int(robot.getBasicTimeStep())
camera.enable(timestep)
led_left = robot.getDevice("front left led")
led_right = robot.getDevice("front right led")
imu = robot.getDevice("inertial unit")
imu.enable(timestep)
gps = robot.getDevice("gps")
gps.enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)

front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")
camera_pitch_motor = robot.getDevice("camera pitch")
camera_pitch_motor.setPosition(1.5)

motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]
for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(1)

robot.current_pose = 6 * [0]  # X, Y, Z, yaw, pitch, roll
robot.target_position = [0, 0, 0]
robot.target_index = 0
robot.target_altitude = 3

time1 = robot.getTime()

roll_disturbance = 0
pitch_disturbance = 0
yaw_disturbance = 0

# determine the targets
target_points = [[-5, 4], [-3, -2], [3, -3], [5, 0], [2, 5]]
target_points = [(x+0.2, y) for (x, y) in target_points]
target_altitude = 2.5


land = False
while robot.step(timestep) != -1:

    roll, pitch, yaw = imu.getRollPitchYaw()
    x_pos, y_pos, altitude = gps.getValues()
    roll_acceleration, pitch_acceleration, _ = gyro.getValues()
    robot.current_pose = [x_pos, y_pos, altitude, roll, pitch, yaw]
    
    if altitude > robot.target_altitude - 1:
        if robot.getTime() - time1 > 0.1:
            yaw_disturbance, pitch_disturbance = move_to_target(robot, target_points)
            time1 = robot.getTime()

    if matches:
        xt, yt = robot.target_position[:2]
        dist = np.sqrt((x_pos-xt)**2 + (y_pos-yt)**2)
        if dist < 0.5:
            land = True
            matches = False

    if land:
        robot.target_altitude -= 0.02
        robot.target_position[2] += 0.01
        robot.target_position[2] = clamp(robot.target_position[2], 0, robot.target_altitude)
        if altitude < 0.08:
            turn_on_led(led_left, led_right)
            print('Landed')
            front_left_motor.setVelocity(0)
            front_right_motor.setVelocity(0)
            rear_left_motor.setVelocity(0)
            rear_right_motor.setVelocity(0)
            break

     
    roll_input = K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
    pitch_input = K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
    yaw_input = yaw_disturbance
    clamped_difference_altitude = clamp(robot.target_altitude - altitude + K_VERTICAL_OFFSET, -1, 1)
    vertical_input = K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

    front_left_motor_input = K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
    front_right_motor_input = K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
    rear_left_motor_input = K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
    rear_right_motor_input = K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input
    
    front_left_motor.setVelocity(front_left_motor_input)
    front_right_motor.setVelocity(-front_right_motor_input)
    rear_left_motor.setVelocity(-rear_left_motor_input)
    rear_right_motor.setVelocity(rear_right_motor_input)

print('Mission Complete')   