from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy

from . util import rotateQuaternion, getHeading
from random import random

from time import time


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = None
        self.ODOM_TRANSLATION_NOISE = None
        self.ODOM_DRIFT_NOISE = None
        self.particlecloud = None
        self.currentPose = None
 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        
       
    def initialise_particle_cloud(self, initialpose):
        pass

 
    
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
        pass
