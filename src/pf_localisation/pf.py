import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from matplotlib.colors import ListedColormap, BoundaryNorm

from .pf_base import PFLocaliserBase
from .util import rotateQuaternion, getHeading

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

    x_prime = m * pos_x
    y_prime = m * pos_y
    return Point(int(x_prime), int(y_prime), 0)


def grid_to_pos(x_prime, y_prime):
    """Convert grid coordinates to position"""
    m = 20

    pos_x = x_prime / m
    pos_y = y_prime / m
    return Point(pos_x, pos_y, 0)


def is_valid(pose, grid):
    """Check if a pose is valid within a given grid"""
    grid_pos = pos_to_grid(pose.position.x, pose.position.y)
    grid_pos.x = clamp(grid_pos.x, 0, 602)
    grid_pos.y = clamp(grid_pos.y, 0, 602)
    return grid[grid_pos.x][grid_pos.y] == 0


def create_grid(grid):
    """Convert grid data to numpy format and reshape"""
    np_grid = np.array(grid.data).flatten()
    return np_grid.reshape(grid.info.width, grid.info.height).transpose()


def get_valid_grid(grid):
    """Get all positions where the grid is 0"""
    np_grid = np.array(grid)
    valid_points = list(zip(*np.where(np_grid == 0)))

    return [grid_to_pos(x, y) for x, y in valid_points]


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


def normalise_list(lst):
    """Normalise a given list between 0 and 1"""
    min_value, max_value = min(lst), max(lst)
    normalized_weights = [(x - min_value) / (max_value - min_value) for x in lst]
    return normalized_weights


def create_cdf(weights):
    """Create a cumulative distribution"""
    num_weights = len(weights)
    cdf = [weights[0]]
    # Each new entry is equivalent to previous weight + current weight
    for i in range(1, num_weights):
        cdf.append(cdf[i - 1] + weights[i])
    cdf = normalise_list(cdf)

    if isDebug:
        plt.plot(cdf)
        plt.xlabel('Index')
        plt.ylabel('Probability')
        plt.title('Cumulative Distribution Function (CDF)')
        plt.grid(True)
        plt.show()

    return cdf


def smooothing_kernel(radius, distance):
    """Generate smoothing from point, given radius"""
    volume = math.pi * (radius ** 8) / 4
    value = max(0, radius ** 2 - distance ** 2)
    return value ** 3 / volume


def clamp(value, minimum, maximum):
    """Clamp a value between -15 and 15."""
    return max(minimum, min(maximum, value))


# --------------------------------------------------------------------- Main Class

class PFLocaliser(PFLocaliserBase):

    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()

        # ----- Set motion model parameters (alpha values)
        self.ODOM_ROTATION_NOISE = 1  # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 1  # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 1  # Odometry model y axis (side-to-side) noise

        self.NOISE_MAX = 5

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 200  # Number of readings to predict

        # ----- Map configuration
        self.grid_map = []
        self.valid_map = []
        self.density_map = []
        self.density_grid = None

    def test_density_function(self):
        pose_array = PoseArray()
        pose_array.poses.append(new_pose(6, 4, 0))
        pose_array.poses.append(new_pose(7, 7, 0))
        pose_array.poses.append(new_pose(3, 0, 0))

        self.particlecloud = pose_array
        self.generate_density_map()

    def density_at_point(self, target_point, particle_array):
        """Get density at given point"""
        # ----- Tuning Parameters
        mass = 1
        smoothing_radius = 10

        density = 0
        for particle in particle_array:
            dx = particle.position.x - target_point.x
            dy = particle.position.y - target_point.y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            influence = smooothing_kernel(smoothing_radius, distance)
            density += mass * influence

        return density

    def generate_density_map(self):
        """Create a particle density map"""
        time_start = time.perf_counter()

        width, height = len(self.grid_map[0]), len(self.grid_map)

        density_matrix = np.zeros((width, height))
        valid_points = np.where(self.grid_map == 0)

        for i, j in zip(*valid_points):
            target_point = grid_to_pos(i, j)
            density_matrix[i, j] = self.density_at_point(target_point, self.particlecloud.poses)

        max_density = np.max(density_matrix)
        if max_density != 0:
            density_matrix /= max_density

        self.density_map = density_matrix

        time_end = time.perf_counter()
        print(f"Generated density map in {time_end - time_start:0.4f} seconds")

        if isDebug:
            plt.imshow(density_matrix, cmap='gray_r')
            plt.colorbar()
            plt.show()

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

        self.grid_map = create_grid(self.occupancy_map)
        self.valid_map = get_valid_grid(self.grid_map)
        # if isDebug:
        # visualize_grid(self.grid_map)
        # self.test_density_function()

        pose_array = PoseArray()
        MAX_RETRIES = 1000
        for _ in range(self.NUMBER_PREDICTED_READINGS):
            insert_pose = Pose()
            retry_count = 0

            while retry_count < MAX_RETRIES:
                deviation_offset = 0.1 + (0.1 * retry_count)

                # Add noise to x, y, and orientation
                noise_x = sample_normal_distribution(deviation_offset) * self.ODOM_TRANSLATION_NOISE
                noise_y = sample_normal_distribution(deviation_offset) * self.ODOM_DRIFT_NOISE
                noise_angle = sample_normal_distribution(deviation_offset) * self.ODOM_ROTATION_NOISE

                insert_pose.position.x = initialpose.pose.pose.position.x + noise_x
                insert_pose.position.y = initialpose.pose.pose.position.y + noise_y
                insert_pose.orientation = rotateQuaternion(initialpose.pose.pose.orientation, noise_angle)

                if is_valid(insert_pose, self.grid_map):
                    break
                retry_count += 1

            pose_array.poses.append(insert_pose)

        return pose_array

    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.

        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

        """
        weights = []
        for pose in self.particlecloud.poses:
            weight = self.sensor_model.get_weight(scan, pose)
            weights.append(weight)

        avg_weight = sum(weights) / len(weights)
        random_particles_count = int(40 * (1 / avg_weight))

        self.ODOM_ROTATION_NOISE = self.NOISE_MAX * (1 / avg_weight)
        self.ODOM_TRANSLATION_NOISE = self.NOISE_MAX * (1 / avg_weight)
        self.ODOM_DRIFT_NOISE = self.NOISE_MAX * (1 / avg_weight)

        resampled_poses = self.systematic_resampling(self.particlecloud.poses, weights, random_particles_count)

        self.particlecloud = resampled_poses

    def systematic_resampling(self, poses, weights, random_particles_count):
        """Resample poses based on weights"""
        M = max(0, len(weights) - random_particles_count)
        cdf = create_cdf(weights)
        # Start in random part of first section
        u = [np.random.uniform(0, 1 / M)]

        S = PoseArray()  # resampled data
        i = 0
        MAX_RETRIES = 1000
        for j in range(0, M):
            # Check if next offset is in next section
            while u[j] > cdf[i]:
                i += 1

            retry_count = 0
            insert_pose = Pose()
            while retry_count < MAX_RETRIES:
                deviation_offset = 0.001 + (0.0001 * retry_count)
                # Random noise for each parameter
                noise_x = sample_normal_distribution(0.01) * self.ODOM_TRANSLATION_NOISE
                noise_y = sample_normal_distribution(0.01) * self.ODOM_DRIFT_NOISE
                noise_angle = sample_normal_distribution(0.01) * self.ODOM_ROTATION_NOISE

                # Add noise to parameter
                insert_pose.position.x = poses[i].position.x + noise_x
                insert_pose.position.y = poses[i].position.y + noise_y
                insert_pose.orientation = rotateQuaternion(poses[i].orientation, noise_angle)

                if is_valid(insert_pose, self.grid_map):
                    break
                retry_count += 1

            S.poses.append(insert_pose)
            u.append(u[j] + 1 / M)

        random_points = np.random.choice(self.valid_map, size=random_particles_count)

        for point in random_points:
            S.poses.append(new_pose(point.x, point.y, random.uniform(0, math.pi * 2)))
        return S

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
