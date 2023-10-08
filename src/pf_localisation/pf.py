import math
import numpy as np
import matplotlib.pyplot as plt

from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from sensor_msgs.msg import LaserScan
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

def laser_scan_setup_test():
    scan = LaserScan()
    scan.header.seq = 22462
    scan.header.stamp.secs = 2246
    scan.header.stamp.nsecs = 300000000
    scan.header.frame_id = "base_laser_link"
    scan.angle_min = -1.5707963705062866
    scan.angle_max = 1.5707963705062866
    scan.angle_increment = 0.006295776925981045
    scan.time_increment = 0.0
    scan.scan_time = 0.0
    scan.range_min = 0.0
    scan.range_max = 5.599999904632568
    scan.ranges = [5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 2.3335647583007812,
             2.2856643199920654, 2.2909138202667236, 2.2428765296936035, 2.248230457305908, 2.253699541091919,
             2.2592849731445312, 2.211060047149658, 2.2167437076568604, 2.2225449085235596, 2.2284653186798096,
             2.234506130218506, 2.186018228530884, 2.1921510696411133, 2.1984057426452637, 2.2047836780548096,
             2.211286783218384, 2.217916488647461, 2.224674701690674, 2.231562614440918, 2.1826179027557373,
             2.1895925998687744, 2.196699380874634, 2.203939914703369, 2.2113163471221924, 2.1619374752044678,
             2.1693954467773438, 2.1769914627075195, 2.184727668762207, 2.192606210708618, 2.200629472732544,
             2.2087996006011963, 2.217118740081787, 3.3969521522521973, 3.410115957260132, 3.5415706634521484,
             3.5556859970092773, 3.570056438446045, 2.0313220024108887, 2.039761781692505, 1.9881075620651245,
             1.996596336364746, 2.0052378177642822, 2.0140345096588135, 2.022989511489868, 2.0321052074432373,
             2.041384696960449, 2.0508310794830322, 2.0604472160339355, 2.0702364444732666, 4.223440647125244,
             4.2440385818481445, 3.437469244003296, 3.4546756744384766, 3.407893657684326, 3.3607685565948486,
             3.313288927078247, 2.089883327484131, 2.1010401248931885, 1.9143632650375366, 1.9248476028442383,
             1.9355244636535645, 1.946398138999939, 1.889973521232605, 1.9008643627166748, 1.9119576215744019,
             1.854569911956787, 1.8656704425811768, 1.8074617385864258, 1.8185573816299438, 1.8298629522323608,
             1.841383457183838, 1.8241733312606812, 1.8129734992980957, 1.8019813299179077, 1.7911922931671143,
             1.7806018590927124, 1.7702056169509888, 1.827691674232483, 1.8172861337661743, 1.8070695400238037,
             1.8635950088500977, 1.8533799648284912, 1.8433488607406616, 1.8334980010986328, 1.8238239288330078,
             1.814322829246521, 1.804991364479065, 1.7958263158798218, 1.7868242263793945, 1.8414815664291382,
             1.8324860334396362, 1.886534571647644, 1.8775556087493896, 1.868735432624817, 1.860071063041687,
             1.8515598773956299, 1.8431987762451172, 1.834985375404358, 1.8269169330596924, 1.81899094581604,
             1.8112047910690308, 1.8636748790740967, 1.8559110164642334, 1.8482847213745117, 1.9001737833023071,
             1.9517210721969604, 1.944027066230774, 1.9951508045196533, 1.9875034093856812, 1.9799925088882446,
             1.972616195678711, 1.9653722047805786, 1.9582585096359253, 1.9512732028961182, 1.9444143772125244,
             1.9376801252365112, 1.9310686588287354, 1.9811835289001465, 1.9746248722076416, 1.9681873321533203,
             1.9618691205978394, 2.011544942855835, 2.005286693572998, 1.9991462230682373, 1.9931219816207886,
             1.9872125387191772, 1.9814163446426392, 1.9757319688796997, 1.9701578617095947, 1.9646928310394287,
             1.9593355655670166, 2.008364677429199, 2.057213068008423, 2.0518910884857178, 2.046677350997925,
             2.0415709018707275, 2.0365703105926514, 2.031674385070801, 2.0268821716308594, 2.0221924781799316,
             2.017604351043701, 2.0131163597106934, 2.061589241027832, 2.0571861267089844, 2.0528831481933594,
             2.0486791133880615, 2.0969979763031006, 2.092886209487915, 2.0888733863830566, 2.08495831489563,
             2.1331686973571777, 2.1293537616729736, 2.125636339187622, 2.122015953063965, 2.1184918880462646,
             2.16664981842041, 2.1632344722747803, 2.1599152088165283, 2.156691312789917, 2.153562307357788,
             2.1505277156829834, 2.1475865840911865, 2.09367299079895, 2.090982675552368, 2.0883820056915283,
             2.0349953174591064, 2.0326313972473145, 2.03035306930542, 1.9774562120437622, 1.9754010438919067,
             1.9734282493591309, 1.9209853410720825, 1.9192225933074951, 1.9175390005111694, 1.9159342050552368,
             1.864028811454773, 1.8626188039779663, 1.8612847328186035, 1.8600261211395264, 1.8086040019989014,
             1.807525873184204, 1.8065204620361328, 1.7554326057434082, 1.7545963525772095, 1.7538303136825562,
             1.7531343698501587, 1.7525084018707275, 1.7519524097442627, 1.701424241065979, 1.7010194063186646,
             1.700682282447815, 1.700412631034851, 1.6001981496810913, 1.6000713109970093, 1.6000078916549683,
             1.6000078916549683, 1.6000714302062988, 1.6001982688903809, 1.550376534461975, 1.5506224632263184,
             1.5509299039840698, 1.551298975944519, 1.5517297983169556, 1.5522223711013794, 1.5026872158050537,
             1.5032838582992554, 1.503940463066101, 1.5046573877334595, 1.5054346323013306, 1.4560633897781372,
             1.4569319486618042, 1.4578593969345093, 1.4588459730148315, 1.459891676902771, 1.4609968662261963,
             1.4117423295974731, 1.4129247665405273, 1.4141653776168823, 1.4154644012451172, 1.416821837425232,
             1.4182382822036743, 1.3690098524093628, 1.3704900741577148, 1.37202787399292, 1.3736237287521362,
             1.3752778768539429, 1.376990556716919, 1.3787623643875122, 1.3294602632522583, 1.331281065940857,
             1.3331598043441772, 1.335096836090088, 1.337092638015747, 1.3391474485397339, 1.341261863708496,
             1.2917656898498535, 1.2939146757125854, 1.2961223125457764, 1.298388957977295, 1.3007153272628784,
             1.303101658821106, 1.3055486679077148, 1.3080568313598633, 1.2582015991210938, 1.2607284784317017,
             1.2633156776428223, 1.2659637928009033, 1.2686734199523926, 1.2714451551437378, 1.2742795944213867,
             1.2239618301391602, 1.2268004417419434, 1.229701042175293, 1.2326643466949463, 1.2356910705566406,
             1.2387819290161133, 1.2419376373291016, 1.2451590299606323, 1.2484467029571533, 1.1973754167556763,
             1.200649380683899, 1.2039891481399536, 1.2073956727981567, 1.210869550704956, 1.214411735534668,
             1.2180231809616089, 1.2217046022415161, 1.2254571914672852, 1.2292817831039429, 1.2331793308258057,
             1.2371509075164795, 1.2411973476409912, 1.2453200817108154, 1.1927235126495361, 1.1968071460723877,
             1.2009665966033936, 1.2052030563354492, 1.2095175981521606, 1.2139114141464233, 1.2183856964111328,
             1.2229417562484741, 1.169124722480774, 1.1736232042312622, 1.1782034635543823, 1.1828665733337402,
             1.1876140832901, 1.192447304725647, 1.1973676681518555, 1.2023766040802002, 1.2074756622314453,
             1.2126665115356445, 1.217950701713562, 1.223329782485962, 1.2288055419921875, 1.2343796491622925,
             1.2400538921356201, 1.2458301782608032, 1.2517104148864746, 1.257696509361267, 1.263790488243103,
             1.2699943780899048, 1.2763103246688843, 1.2827404737472534, 1.2892870903015137, 1.295952558517456,
             1.302739143371582, 1.3096493482589722, 1.3166857957839966, 1.3238509893417358, 1.3311476707458496,
             1.3385785818099976, 1.346146583557129, 1.6246254444122314, 3.8127756118774414, 5.478811740875244,
             5.5113983154296875, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568, 5.599999904632568,
             5.599999904632568, 5.523097991943359, 5.5060505867004395, 5.378429412841797, 5.25178861618042,
             5.236343860626221, 5.22119665145874, 5.04193115234375, 5.027826309204102, 4.959497928619385,
             4.837385654449463, 4.8245344161987305, 4.703808307647705, 4.691747665405273, 4.679934024810791,
             4.668364524841309, 4.496448516845703, 4.485741138458252, 4.475261688232422, 4.465007305145264,
             4.401940822601318, 4.339328289031982, 4.118747234344482, 4.11004114151001, 4.101534366607666,
             4.09322452545166, 4.032736778259277, 3.8681023120880127, 3.80859637260437, 3.8015401363372803,
             3.74267840385437, 3.736064910888672, 3.7296223640441895, 3.6716361045837402, 3.613987445831299,
             3.6082143783569336, 3.5511364936828613, 3.545762062072754, 3.489231824874878, 3.4842422008514404,
             3.428236961364746, 3.423619508743286, 3.368117570877075, 3.3128914833068848, 3.3088390827178955,
             3.304927349090576, 3.250368595123291, 3.1960608959198, 3.192674160003662, 3.1894209384918213,
             3.1357243061065674, 3.1327836513519287, 3.079488754272461, 3.0768494606018066, 3.074336528778076,
             3.021589517593384, 2.9690420627593994, 2.966974973678589, 2.9650282859802246, 2.9632015228271484,
             2.9614944458007812, 2.9599063396453857, 2.908294439315796, 2.9069666862487793, 2.9057555198669434,
             2.90466046333313, 2.9036812782287598, 2.852768898010254, 2.852033853530884, 2.801387310028076,
             2.8008875846862793, 2.800499200820923, 2.7502176761627197, 2.750054359436035, 2.75]
    scan.intensities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    return scan


# --------------------------------------------------------------------- Main Class

class PFLocaliser(PFLocaliserBase):

    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()

        # ----- Set motion model parameters (alpha values)
        self.ODOM_ROTATION_NOISE = 3  # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 2  # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 7  # Odometry model y axis (side-to-side) noise

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
            noise_x = sample_normal_distribution(0.3) * self.ODOM_TRANSLATION_NOISE  # 0.3 is the variance
            noise_y = sample_normal_distribution(0.3) * self.ODOM_DRIFT_NOISE  # 0.3 is the variance
            noise_angle = sample_normal_distribution(0.3) * self.ODOM_TRANSLATION_NOISE  # 0.3 is the variance

            position_x = initialpose.pose.pose.position.x + noise_x  # need to multiply by parameter
            position_y = initialpose.pose.pose.position.y + noise_y  # need to multiply by parameter
            orientation = rotateQuaternion(initialpose.pose.pose.orientation, noise_angle)  # need to multiply by parameter

            pose_array.poses.append(new_pose(position_x, position_y, orientation))

        self.particlecloud = pose_array
        return pose_array

    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.

        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        particle_num = len(self.particlecloud.poses)
        new_cloud = []

        for i in range(0, particle_num):
            weight = self.sensor_model.get_weight(scan, self.particlecloud.poses[i])
            new_cloud.append(weight)

        self.particleCloud = new_cloud
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
    laser_scan = laser_scan_setup_test()
    localiser.update_particle_cloud(laser_scan)


if __name__ == "__main__":
    main()
