import numpy as np
import polytope as pc
from src.PolyUtils import PolyUtils
from src.ReachtubeSegment import ReachtubeSegment
from src.Agent import Agent, DRONE_TYPE
from typing import List, Dict, Tuple
from scipy.spatial import ConvexHull
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
import mpl_toolkits.mplot3d as plt3d
import itertools
from mayavi import mlab
from src.Agent import UniformTubeset


class Plotter:
    '''
    colors = [(int(141/255),int(211/255),int(199/255)), (int(255/255),int(255/255),int(179/255)), (int(190/255),int(186/255),int(218/255)),
    (int(251/255),int(128/255),int(114/255)), (int(128/255),int(177/255),int(211/255)), (int(253/255),int(180/255),int(98/255)), (int(179/255),int(222/255),int(105/255)),
              (int(252/255),int(205/255),int(229/255)), (int(217/255),int(217/255),int(217/255)),
              (int(188/255), int(128/255),int(189/255)), (int(204/255),int(235/255),int(197/255)), (int(255/255),int(237/255),int(111/255))]
    '''
    # colors = ['cool', 'summer', 'purples', 'spring', 'gist_earth', 'spectral', 'blues']
    colors = ['#b3de69',  '#bebada', '#fdb462', '#fccde5', '#bc80bd', '#ccebc5', '#ffed6f', '#ffffb3', '#c7eae5']

    colors = [
        (179/255, 222/255, 105/255),
        (190/255, 186/255, 218/255),
        (255/255, 255/255, 179/255),
        (128/255, 177/255, 211/255),
        (253/255, 180/255, 98/255),
        (252/255, 205/255, 229/255),
        (188/255, 128/255, 189/255),
        (204/255, 235/255, 197/255),
        (255/255, 237/255, 111/255)
    ]

    # TODO move away from static plotting and move towards OO plotting configurations
    @staticmethod
    def verts_to_tri():
        tri = np.row_stack([(0, 1, 2), (1, 2, 3), (0, 1, 4), (1, 4, 5), (0, 2, 4), (2, 4, 6), (3, 5, 7), (1, 3, 5),
                            (3, 6, 7), (2, 3, 6), (5, 6, 7), (4, 5, 6)])
        return tri

    @staticmethod
    def hrect_to_tri(hrect):
        x = [hrect[0][0]] * 4 + [hrect[1][0]] * 4
        y = [hrect[0][1]] * 2 + [hrect[1][1]] * 2
        y *= 2
        z = [hrect[0][2]] + [hrect[1][2]]
        z *= 4
        return Plotter.verts_to_tri(), x, y, z

    @staticmethod
    def cube_faces(xmin, xmax, ymin, ymax, zmin, zmax):
        faces = []

        x, y = np.mgrid[xmin:xmax:3j, ymin:ymax:3j]
        z = np.ones(y.shape) * zmin
        faces.append((x, y, z))

        x, y = np.mgrid[xmin:xmax:3j, ymin:ymax:3j]
        z = np.ones(y.shape) * zmax
        faces.append((x, y, z))

        x, z = np.mgrid[xmin:xmax:3j, zmin:zmax:3j]
        y = np.ones(z.shape) * ymin
        faces.append((x, y, z))

        x, z = np.mgrid[xmin:xmax:3j, zmin:zmax:3j]
        y = np.ones(z.shape) * ymax
        faces.append((x, y, z))

        y, z = np.mgrid[ymin:ymax:3j, zmin:zmax:3j]
        x = np.ones(z.shape) * xmin
        faces.append((x, y, z))

        y, z = np.mgrid[ymin:ymax:3j, zmin:zmax:3j]
        x = np.ones(z.shape) * xmax
        faces.append((x, y, z))

        return faces

    @staticmethod
    def mlab_plt_cube(xmin, xmax, ymin, ymax, zmin, zmax, colormap, opacity = 0.5):
        faces = Plotter.cube_faces(xmin, xmax, ymin, ymax, zmin, zmax)
        for grid in faces:
            x, y, z = grid
            mlab.mesh(x, y, z, opacity=opacity,  color=colormap)

    @staticmethod
    @mlab.show
    def plot(drone_num, tubes: List[List[ReachtubeSegment]], agent_trace, agents_list: List[Agent], unsafe_set, init_set = [], goal_set = [], sym_level=0,
             plot_step=10):
        # agents_tubes = [[segment.tube for segment in segment_list] for segment_list in tubes]
        colors = ['b', 'g', 'y']

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.set_aspect("equal")
        waypointslists = [[waypoint.mode_parameters for waypoint in agent.mode_list] for agent in agents_list]
        legend_drone_rect = []
        legend_unsafe_set = []

        for d in range(drone_num):
            curr_agent = agents_list[d]
            color = colors[d]
            # wp = np.array(waypointslists[d])
            # if d == 0:
            #     #plt.scatter(wp[:, 0], wp[:, 1], 'k' + 'o', label='waypoints')
            #     mlab.plot3d(wp[:, 0], wp[:, 1], wp[:, 2])
            #     #ax.add_line(line)
            #     #ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], 'k')
            #     # plt.scatter(wp[:, 0], wp[:, 1], 'k')
            # else:
            #     #plt.scatter(wp[:, 0], wp[:, 1], 'k' + 'o')
            #     mlab.plot3d(wp[:, 0], wp[:, 1], wp[:, 2])
            #     #ax.add_line(line)
            #     #ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], 'k')
            wp = waypointslists[d]
            for i in range(1,len(wp)):
                mlab.plot3d([wp[i][0],wp[i][3]],[wp[i][1],wp[i][4]],
                    [wp[i][2],wp[i][5]], color = (0,0,0))

            curr_tubes = tubes[d]
            for ci in range(len(curr_tubes)):
                curr_segment = curr_tubes[ci]
                curr_tube_list = curr_segment.tube_list
                traces = curr_segment.trace
                for curr_tube in curr_tube_list:
                    if sym_level == 2:
                        colormap = Plotter.colors[curr_segment.virtual_mode]
                    else:
                        colormap = Plotter.colors[0]
                    for i in range(0, len(curr_tube), plot_step):
                        box_of_poly = PolyUtils.get_bounding_box(curr_tube[i])
                        Plotter.mlab_plt_cube(box_of_poly[0, 0], box_of_poly[1, 0], box_of_poly[0, 1], box_of_poly[1, 1], box_of_poly[0, 2], box_of_poly[1, 2], colormap = colormap, opacity = 1.0)

            color = (217/255, 217/255, 217/255)
            for i in range(1,len(unsafe_set)):
                box_of_poly = np.array(pc.bounding_box(unsafe_set[i].unsafe_set))
                Plotter.mlab_plt_cube(box_of_poly[0, 0], box_of_poly[1, 0], box_of_poly[0, 1], box_of_poly[1, 1],
                                      box_of_poly[0, 2], box_of_poly[1, 2], colormap=color, opacity = 0.1)

            color = (128/255,177/255,211/255)
            for i in range(len(init_set)):
                reg = init_set[i]
                box_of_poly = np.array(pc.bounding_box(reg))
                Plotter.mlab_plt_cube(box_of_poly[0, 0], box_of_poly[1, 0], box_of_poly[0, 1], box_of_poly[1, 1],
                                      box_of_poly[0, 2], box_of_poly[1, 2], colormap=color, opacity = 0.8)

            color = (141/255,211/255,199/255)
            for i in range(len(goal_set)):
                reg = goal_set[i]
                box_of_poly = np.array(pc.bounding_box(reg))
                Plotter.mlab_plt_cube(box_of_poly[0, 0], box_of_poly[1, 0], box_of_poly[0, 1], box_of_poly[1, 1],
                                      box_of_poly[0, 2], box_of_poly[1, 2], colormap=color, opacity=0.8)

            # mlab.xlabel('x')
            # mlab.ylabel('y')
            # mlab.zlabel('z')
            mlab.show()
            '''
            tri = None
            x = []
            y = []
            z = []
            # print("boundedboxed final tube:")
            for ci in range(len(curr_tubes)):
                curr_segment = curr_tubes[ci]
                curr_tube = curr_segment.tube
                traces = curr_segment.trace
                for i in range(0, len(curr_tube), plot_step):
                    box_of_poly = PolyUtils.get_bounding_box(curr_tube[i])
                    _triangles, _x, _y, _z = Plotter.hrect_to_tri(box_of_poly)
                    if tri is None:
                        tri = _triangles
                    else:
                        tri = np.row_stack((tri, _triangles))
                    z.extend(_z)
                    x.extend(_x)
                    y.extend(_y)

                for i in range(0, len(curr_tube), plot_step):
                    box_of_poly = PolyUtils.get_bounding_box(curr_tube[i])

                    rect = Rectangle(box_of_poly[0, 0:2], box_of_poly[1, 0] - box_of_poly[0, 0],
                                     box_of_poly[1, 1] - box_of_poly[0, 1], linewidth=1,
                                     edgecolor=color, facecolor='none')
                    if ci == 0 and i == 0:
                        legend_drone_rect.append(rect)
                    ax.add_patch(rect)
                '''
            #triplot = mlab.triangular_mesh(x, y, z, list(tri), colormap='summer')
        # color = 'r'
        '''
        if isinstance(unsafe_set, np.ndarray):
            unsafe_set = pc.box2poly(unsafe_set.T)
        if isinstance(unsafe_set, pc.Polytope):
            unsafe_set = [unsafe_set]
        for i in range(len(unsafe_set)):
            poly = unsafe_set[i]
            points = pc.extreme(poly)
            points = points[:, :2]
            hull = ConvexHull(points)
            poly_patch = Polygon(points[hull.vertices, :], alpha=.5, color=color, fill=True)
            if i == 0:
                legend_unsafe_set.append(poly_patch)
            ax.add_patch(poly_patch)
        '''
        # plt.legend((legend_drone_rect[0], legend_drone_rect[1], legend_unsafe_set[0]), ("first drone tube", "second drone tube", "unsafe set"))
        '''plt.ylim([-10, 15])
        plt.xlim([-5, 25])
        plt.tick_params(axis='both', which='major', labelsize=30)
        '''

    @staticmethod
    @mlab.show
    def plot_virtual(tube_dict: Dict[Tuple[float, ...], List[np.array]], agents_list: List[Agent], abs_nodes_neighbors,
                     plot_step=1):
        # agents_tubes = [[segment.tube for segment in segment_list] for segment_list in tubes]
        colors = ['b', 'g', 'y']

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.set_aspect("equal")
        # waypointslists = [[waypoint.mode_parameters for waypoint in agent.mode_list] for agent in agents_list]
        legend_drone_rect = []
        legend_unsafe_set = []

        # agents_tubes = [[segment.tube for segment in segment_list] for segment_list in tubes]
        # waypointslists = [[waypoint.mode_parameters for waypoint in agent.abs_mode_list] for agent in agents_list]
        # wp = np.array(waypointslists[0])
        # mlab.plot3d(wp[:, 0], wp[:, 1], wp[:, 2])
        # legend_drone_rect = []
        # legend_unsafe_set = []

        color_index = 0
        for virtual_mode in tube_dict:
            curr_tube_list = tube_dict[virtual_mode]
            colormap = Plotter.colors[virtual_mode]

            for curr_tube in curr_tube_list:
                for i in range(0, len(curr_tube), plot_step):
                    box_of_poly = curr_tube[i]  # PolyUtils.get_region_bounding_box(curr_tube[i])
                    Plotter.mlab_plt_cube(box_of_poly[0, 0], box_of_poly[1, 0], box_of_poly[0, 1],
                                          box_of_poly[1, 1], box_of_poly[0, 2], box_of_poly[1, 2], colormap=colormap)

        mlab.xlabel('x')
        mlab.ylabel('y')
        mlab.zlabel('z')
        mlab.show()
