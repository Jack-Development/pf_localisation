from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy
import numpy as np

from . util import rotateQuaternion, getHeading
from random import random

from time import time


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters (alpha values)
        self.ODOM_ROTATION_NOISE = None # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = None # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = None # Odometry model y axis (side-to-side) noise

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20 # Number of readings to predict

        # ----- Particle cloud configuration
        self.NUMBER_OF_PARTICLES = 200
        
       
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
        pose_array = PoseArray()
        for _ in range(self.NUMBER_OF_PARTICLES):
            pose_to_append = Pose()

            noise = sample_normal_distribution(0.3) # 0.3 is the variance
            pose_to_append.position.x = initialpose.pose.pose.position.x + noise # need to multiply by parameter

            noise = sample_normal_distribution(0.3) # 0.3 is the variance
            pose_to_append.position.y = initialpose.pose.pose.position.y +  noise # need to multiply by parameter

            noise = sample_normal_distribution(0.3) # 0.3 is the variance
            pose_to_append.orientation = rotateQuaternion(initialpose.pose.pose.orientation, noise) # need to multiply by parameter
            
            pose_array.poses.append(pose_to_append)
        print(pose_array)
        return pose_array

 
    
    def update_particle_cloud(self, scan):
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
        pass

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