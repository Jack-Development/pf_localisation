from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from . pf_base import PFLocaliserBase
import math
import rospy

from . util import rotateQuaternion, getHeading
from random import random

from time import time


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
        
        # ----- Set motion model parameters
 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        
       
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
        pass

 
    
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

