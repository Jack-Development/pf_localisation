from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from . pf_base import PFLocaliserBase
import math
import rospy
import numpy as np

from .util import rotateQuaternion, getHeading
from random import random

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def visualize_grid(grid):
    """
    Visualize a 2D array where:
    - 0 values are white
    - 100 values are black
    - -1 values are red (representing unknown)

    Parameters:
    - grid (list of list of int): 2D array containing values between -1 and 100.
    """

    # Create a custom colormap: -1 is red, 0 is white, and 100 is black
    colors = [(1, 0, 0), (1, 1, 1), (0, 0, 0)]  # R -> W -> B
    cmap = ListedColormap(colors)

    # Create a custom norm to map -1 to red, 0 to white, and 100 to black
    bounds = [-1.5, -0.5, 0.5, 100.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # Plot the grid
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[-1, 0, 100], label="Value")
    plt.title("Visualization of 2D Array")
    plt.show()


def pos_to_grid(pos_x, pos_y):
    m = 20
    c = 300

    x_prime = m * pos_x + c
    y_prime = -m * pos_y + c
    return x_prime, y_prime


def is_valid(pose, grid):
    print(pose)
    grid_pos = pos_to_grid(pose.position.x, pose.position.y)
    print(grid_pos)

    return grid[grid_pos[0]][grid_pos[1]] == 0


def create_grid(grid):
    np_grid = np.array(grid.data).flatten()
    np_grid = np_grid.reshape(grid.info.width, grid.info.height)
    return np_grid.transpose()


# Position:
# X: (.position.x)
# Y: (.position.y)

# Orientation:
# z:
# w:
# ----- Utility, returns new pose with given attributes
def new_pose(x, y, angle):
	
	pose = Pose()
	
	point = Point()
	point.x = x
	point.y = y
	pose.position = point
	
	quaternion = Quaternion()
	quaternion.w = 1.0
	quaternion = rotateQuaternion(quaternion, angle)
	pose.orientation = quaternion
	
	return pose


class PFLocaliser(PFLocaliserBase):

    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters (alpha values)
        self.ODOM_ROTATION_NOISE = 3 # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 2 # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 7 # Odometry model y axis (side-to-side) noise

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20 # Number of readings to predict

        # ----- Particle cloud configuration
        self.NUMBER_OF_PARTICLES = 200
        
        # ----- Set initial grid map for validity checking
        self.grid_map = []
        
    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        grid_map = create_grid(self.occupancy_map)
        print(is_valid(initialpose.pose.pose, grid_map))

        visualize_grid(grid_map)
      
        pose_array = PoseArray()
        for _ in range(self.NUMBER_OF_PARTICLES):
            pose_to_append = Pose()

            noise = sample_normal_distribution(0.3)*self.ODOM_TRANSLATION_NOISE # 0.3 is the variance
            pose_to_append.position.x = initialpose.pose.pose.position.x + noise # need to multiply by parameter

            noise = sample_normal_distribution(0.3)*self.ODOM_DRIFT_NOISE # 0.3 is the variance
            pose_to_append.position.y = initialpose.pose.pose.position.y +  noise # need to multiply by parameter

            noise = sample_normal_distribution(0.3)*self.ODOM_TRANSLATION_NOISE # 0.3 is the variance
            pose_to_append.orientation = rotateQuaternion(initialpose.pose.pose.orientation, noise) # need to multiply by parameter
            
            pose_array.poses.append(pose_to_append)
        return pose_array

    def update_particle_cloud(self, scan):
        new_scan = sensor_msgs.msg.LaserScan
        particleNo = 1

        """Step 1 of particle filter algorithm"""
        new_cloud = []

        """Step 2"""
        for i in range(0,particleNo):
            """Step 3"""
            """Step 4"""
            weight = self.sensor_model.get_weight(self,scan,self.particlecloud[i])
            """Step 5"""
            new_cloud.append(weight)
            print(new_cloud)
        self.particleCloud = new_cloud

        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        pass

    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """        
        # ----- Basic implementation, returns mean pose of all particles
        
        particles = len(self.particlecloud.poses)
        
        x_sum = 0
        y_sum = 0
        sin_angle_sum = 0
        cos_angle_sum = 0
        
        for i in range(0, particles):
            
            x_sum += self.particlecloud.poses[i].position.x
            y_sum += self.particlecloud.poses[i].position.y
            
            angle = getHeading(self.particlecloud.poses[i].orientation)
            sin_angle_sum += math.sin(angle)
            cos_angle_sum += math.cos(angle)
        
        x_mean = x_sum / particles
        y_mean = y_sum / particles
        angle_mean = math.atan2(sin_angle_sum, cos_angle_sum)
        
        return new_pose(x_mean, y_mean, angle_mean)

# sampling
def sample_normal_distribution(variance):
    s = np.random.normal(0, math.sqrt(variance))
    return s


# for debugging
def main():
    localiser = PFLocaliser()
    localiser.initialise_particle_cloud(Pose())

if __name__ == "__main__":
    main()
