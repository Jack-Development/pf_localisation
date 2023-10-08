import math
import numpy as np
import matplotlib.pyplot as plt

from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from matplotlib.colors import ListedColormap, BoundaryNorm

from . pf_base import PFLocaliserBase
from . util import rotateQuaternion, getHeading

""" Enable for debug functions """
isDebug = False


# --------------------------------------------------------------------- Utility Functions

def visualize_grid(grid):
    """
    Visualize a 2D array with:
    - 0 values as white
    - 100 values as black
    - -1 values as red (representing unknown)
    """
    colors = [(1, 0, 0), (1, 1, 1), (0, 0, 0)]  # Red -> White -> Black
    cmap = ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 100.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[-1, 0, 100], label="Value")
    plt.title("Visualization of 2D Array")
    plt.show()


def pos_to_grid(pos_x, pos_y):
    """Convert position to grid coordinates"""
    m = 20
    c = 300

    x_prime = m * pos_x + c
    y_prime = -m * pos_y + c
    return x_prime, y_prime


def is_valid(pose, grid):
    """Check if a pose is valid within a given grid"""
    grid_pos = pos_to_grid(pose.position.x, pose.position.y)
    return grid[grid_pos[0]][grid_pos[1]] == 0


def create_grid(grid):
    """Convert grid data to numpy format and reshape"""
    np_grid = np.array(grid.data).flatten()
    return np_grid.reshape(grid.info.width, grid.info.height).transpose()


def sample_normal_distribution(variance):
    """Sample from a normal distribution"""
    return np.random.normal(0, math.sqrt(variance))


def new_pose(x, y, angle):
    """Create a new Pose with given coordinates and angle"""
    pose = Pose()
    pose.position = Point(x, y, 0)

    if type(angle) is Quaternion:
        pose.orientation = angle
    else:
        quaternion = Quaternion(w=1.0)
        pose.orientation = rotateQuaternion(quaternion, angle)

    return pose


# --------------------------------------------------------------------- Main Class

class PFLocaliser(PFLocaliserBase):

    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()

        # ----- Set motion model parameters (alpha values)
        self.ODOM_ROTATION_NOISE = 1  # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 1  # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 1  # Odometry model y axis (side-to-side) noise

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20  # Number of readings to predict

        # ----- Particle cloud configuration
        self.NUMBER_OF_PARTICLES = 200
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

        if isDebug:
            visualize_grid(grid_map)

        pose_array = PoseArray()
        for _ in range(self.NUMBER_OF_PARTICLES):
            # Add noise to x, y, and orientation
            noise_x = sample_normal_distribution(0.1) * self.ODOM_TRANSLATION_NOISE
            noise_y = sample_normal_distribution(0.1) * self.ODOM_DRIFT_NOISE
            noise_angle = sample_normal_distribution(0.1) * self.ODOM_TRANSLATION_NOISE

            position_x = initialpose.pose.pose.position.x + noise_x
            position_y = initialpose.pose.pose.position.y + noise_y
            orientation = rotateQuaternion(initialpose.pose.pose.orientation, noise_angle)

            pose_array.poses.append(new_pose(position_x, position_y, orientation))

        return pose_array

    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.

        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        new_cloud = []

        for pose in self.particlecloud.poses:
            weight = self.sensor_model.get_weight(scan, pose)
            new_cloud.append(weight)

        self.particleCloud = new_cloud

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

        x_sum, y_sum, sin_angle_sum, cos_angle_sum = 0, 0, 0, 0

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


# --------------------------------------------------------------------- Debugging Functions
def main():
    """Start example localiser and test particle_cloud"""
    localiser = PFLocaliser()
    localiser.initialise_particle_cloud(new_pose(10, 5, 0))


if __name__ == "__main__":
    main()
