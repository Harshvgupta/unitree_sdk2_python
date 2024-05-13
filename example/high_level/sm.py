import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import threading
import math
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactortyInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)

class RobotController:
    def __init__(self):
        self.client = SportClient() 
        self.client.SetTimeout(10.0)
        self.client.Init()
        self.initial_pos = np.array([0.01687752828001976, -0.015443671494722366, 0.303792268037796])  # Initial position
        self.current_pos = self.initial_pos.copy()  # Current position
        self.target_pos_camera = np.array([-0.18, 0.33, 3.40])  # Example target position in camera frame
        self.dt = 0.1  # Time step for velocity update (increased to 0.1 seconds)
        self.max_velocity = 0.2  # Maximum velocity limit
        self.min_distance = 0.1  # Minimum distance to target
        self.positions = [self.initial_pos[:2]]  # Initialize positions with the initial position
        self.velocities = []
        self.yaw = 0  # Initial yaw angle
        self.velocity_thread = None
        self.stop_velocity_thread = False

    def calculate_velocity(self):
        # Transform the target position from the camera frame to the robot frame
        target_pos_robot = self.transform_camera_to_robot(self.target_pos_camera)

        vector_to_target = target_pos_robot - self.current_pos
        horizontal_distance = np.linalg.norm(vector_to_target[:2])

        if horizontal_distance < self.min_distance:
            return np.zeros(3)  # Close enough to target

        direction = vector_to_target[:2] / horizontal_distance
        speed = min(horizontal_distance, self.max_velocity)  # Adjust speed based on distance
        vx = np.clip(direction[0] * speed, -self.max_velocity, self.max_velocity)
        vy = np.clip(direction[1] * speed, -self.max_velocity, self.max_velocity)

        # Calculate the desired yaw angle to face the target position
        desired_yaw = math.atan2(target_pos_robot[1] - self.current_pos[1], target_pos_robot[0] - self.current_pos[0])
        yaw_error = desired_yaw - self.yaw

        # Adjust the yaw error to be within -pi to pi range
        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        # Calculate the yaw velocity based on the yaw error
        yaw_velocity = self.calculate_yaw_velocity(yaw_error)

        return np.array([vx, vy, yaw_velocity])

    def transform_camera_to_robot(self, camera_pos):
        R = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        return R.dot(camera_pos)

    def update_position(self, velocities):
        self.current_pos[:2] += velocities[:2] * self.dt
        self.yaw += velocities[2] * self.dt  # Update yaw angle
        self.positions.append(self.current_pos[:2])  # Append current position to positions list

    def calculate_yaw_velocity(self, yaw_error, k_p=1.0):
        # Calculate the yaw velocity using a proportional controller
        yaw_velocity = k_p * yaw_error

        # Limit the yaw velocity to a maximum value
        max_yaw_velocity = 0.5  # Adjust this value based on your robot's constraints
        yaw_velocity = np.clip(yaw_velocity, -max_yaw_velocity, max_yaw_velocity)

        return yaw_velocity

    def velocity_update_thread(self):
        while not self.stop_velocity_thread:
            target_pos_robot = self.transform_camera_to_robot(self.target_pos_camera)
            distance_to_target = np.linalg.norm(target_pos_robot[:2] - self.current_pos[:2])

            if distance_to_target < self.min_distance:
                break

            # Calculate the velocity based on the current position and target position
            velocities = self.calculate_velocity()

            # Send velocity command to the robot for a short duration using a for loop
            elapsed_time = 1  # Adjust the duration as needed
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(velocities[0], velocities[1], velocities[2])
                print(velocities[0], velocities[1], velocities[2])
                time.sleep(self.dt)

                # Update the current position and yaw based on the robot's feedback
                self.GetInitState(robot_state)

                # Recalculate the distance to the target
                distance_to_target = np.linalg.norm(target_pos_robot[:2] - self.current_pos[:2])

                # Check if the target position is reached
                if distance_to_target < self.min_distance:
                    break

            # Stop the robot's movement
            self.client.StopMove()

            # Update the position and velocities lists
            self.update_position(velocities)
            self.velocities.append(velocities)
    
    def run(self):
        print("Start test !!!")

        # Start the velocity update thread
        self.velocity_thread = threading.Thread(target=self.velocity_update_thread)
        self.velocity_thread.start()

        while True:
            # Check if the target position is reached
            target_pos_robot = self.transform_camera_to_robot(self.target_pos_camera)
            if np.linalg.norm(target_pos_robot[:2] - self.current_pos[:2]) < self.min_distance:
                break

            time.sleep(0.1)  # Wait for a short duration before checking again

        # Stop the velocity update thread
        self.stop_velocity_thread = True
        self.velocity_thread.join()

        print("Target reached. Stopping.")
        self.client.StopMove()  # Stop the robot's movement

        self.visualize()

    def visualize(self):
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Robot path')
        plt.plot(self.initial_pos[0], self.initial_pos[1], 'go', label='Initial position')  # Plot initial position
        target_pos_robot = self.transform_camera_to_robot(self.target_pos_camera)
        plt.plot(target_pos_robot[0], target_pos_robot[1], 'ro', label='Target position')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Robot Path')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        plt.subplot(1, 2, 2)
        plt.plot(velocities[:, 0], label='vx')
        plt.plot(velocities[:, 1], label='vy')
        plt.plot(velocities[:, 2], label='yaw_rate')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Velocity Profile')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def GetInitState(self, robot_state: SportModeState_):
        self.current_pos[0] = robot_state.position[0]
        self.current_pos[1] = robot_state.position[1]
        self.yaw = robot_state.imu_state.rpy[2]

robot_state = unitree_go_msg_dds__SportModeState_()

def HighStateHandler(msg: SportModeState_):
    global robot_state
    robot_state = msg

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactortyInitialize(0, sys.argv[1])
    else:
        ChannelFactortyInitialize(0)

    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init(HighStateHandler, 10)
    time.sleep(1)

    robot_controller = RobotController()
    robot_controller.run()