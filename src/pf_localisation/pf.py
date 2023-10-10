import math
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
    c = 300

    x_prime = m * pos_x + c
    y_prime = -m * pos_y + c
    return Point(int(x_prime), int(y_prime), 0)


def grid_to_pos(x_prime, y_prime):
    """Convert grid coordinates to position"""
    m = 20
    c = 300

    pos_x = (x_prime - c) / m
    pos_y = (c - y_prime) / m
    return Point(pos_x, pos_y, 0)


def is_valid(pose, grid):
    """Check if a pose is valid within a given grid"""
    grid_pos = pos_to_grid(pose.position.x, pose.position.y)
    return grid[grid_pos.x][grid_pos.y] == 0


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


def systematic_resampling(poses, weights):
    """Resample poses based on weights"""
    M = len(weights)
    cdf = create_cdf(weights)
    # Start in random part of first section
    u = [np.random.uniform(0, 1 / M)]

    S = PoseArray()  # resampled data
    i = 0
    for j in range(0, M):
        # Check if next offset is in next section
        while u[j] > cdf[i]:
            i += 1

        # Random noise for each parameter
        noise_x = sample_normal_distribution(0.001)  # need to multiply by parameter
        noise_y = sample_normal_distribution(0.001)  # need to multiply by parameter
        noise_angle = sample_normal_distribution(0.001)  # need to multiply by parameter

        # Add noise to parameter
        position_x = poses[i].position.x + noise_x
        position_y = poses[i].position.y + noise_y
        orientation = rotateQuaternion(poses[i].orientation, noise_angle)

        S.poses.append(new_pose(position_x, position_y, orientation))
        u.append(u[j] + 1 / M)
    return S


def smooothing_kernel(radius, distance):
    """Generate smoothing from point, given radius"""
    volume = math.pi * (radius ** 8) / 4
    value = max(0, radius ** 2 - distance ** 2)
    return value ** 3 / volume


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
        if isDebug:
            visualize_grid(self.grid_map)
            # self.test_density_function()

        pose_array = PoseArray()
        for i in range(self.NUMBER_OF_PARTICLES):
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
        weights = []
        for pose in self.particlecloud.poses:
            weight = self.sensor_model.get_weight(scan, pose)
            weights.append(weight)

        resampled_poses = systematic_resampling(self.particlecloud.poses, weights)

        self.particlecloud = resampled_poses

    def estimate_pose(self):
        """
        The map is of size 602*602
        We will break it up into an 86*86 grid (squares of side length 7)
        """
        SQUARE_SIDE_LENGTH = 1
        NUMBER_OF_SQUARES = int(602/SQUARE_SIDE_LENGTH)
        heatmap = np.zeros((NUMBER_OF_SQUARES, NUMBER_OF_SQUARES, 2))
        for pose in self.particlecloud.poses:
            x = math.floor(pose.position.x) - 15 + NUMBER_OF_SQUARES//2
            y = math.floor(pose.position.y) - 15 + NUMBER_OF_SQUARES//2

            angle = getHeading(pose.orientation)
            sin_angle = math.sin(angle)
            cos_angle = math.cos(angle)
            angle = math.atan2(sin_angle, cos_angle)
            heatmap[x, y, 0] += angle

            heatmap[x,y,1] += 1 # represents the number of particles in a specific square of the heatmap
        
        if isDebug:
            plt.imshow(heatmap, cmap='viridis', interpolation='nearest', vmin=0, vmax=86)
            plt.xlim(0, 86)
            plt.colorbar()
            plt.show()

        max_index = np.argmax(heatmap[:,:,1])
        max_index_2d = np.unravel_index(max_index, heatmap[:,:,1].shape)

        x = max_index_2d[0]
        y = max_index_2d[1]

        angle_mean = float(heatmap[x, y, 0])/float(heatmap[x, y, 1])

        # convert position in heatmap to real position - this should be more granular
        x = (x - NUMBER_OF_SQUARES // 2) + 15
        y = (y - NUMBER_OF_SQUARES // 2) + 15


        newpose = new_pose(x,y,angle_mean)
        return newpose

# --------------------------------------------------------------------- Debugging Functions
def main():
    """Start example localiser and test particle_cloud"""
    localiser = PFLocaliser()
    localiser.initialise_particle_cloud(new_pose(10, 5, 0))


if __name__ == "__main__":
    main()
