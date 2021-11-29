import numpy as np
import polytope as pc
from src.PolyUtils import PolyUtils
from src.ReachtubeSegment import ReachtubeSegment
from src.Agent import Agent, DRONE_TYPE
from typing import List, Dict, Tuple
from scipy.spatial import ConvexHull
import matplotlib
# matplotlib.use("macOSX")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.lines as mlines
from src.Agent import UniformTubeset
import pdb
import pypoman as ppm

class Plotter:
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g']
    # colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5']
    # colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
    colors = ['#b3de69',  '#bebada', '#fdb462', '#fccde5', '#bc80bd', '#ccebc5', '#ffed6f', '#ffffb3', '#c7eae5']
    # colors = ['#8dd3c7', '#8dd3c7', '#8dd3c7', '#8dd3c7', '#8dd3c7', '#8dd3c7', '#8dd3c7', '#8dd3c7', '#8dd3c7', '#8dd3c7', '#8dd3c7', '#8dd3c7']
    # TODO move away from static plotting and move towards OO plotting configurations
    @staticmethod
    def plot(drone_num, tubes: List[List[ReachtubeSegment]], traces: List[List[np.array]], agents_list: List[Agent],  unsafe_set, init_set = [], goal_set = [], sym_level = 0, plot_step=1):
        # agents_tubes = [[segment.tube for segment in segment_list] for segment_list in tubes]
        # pdb.set_trace()
        plt.figure()
        currentAxis = plt.gca()
        # waypointslists = [[waypoint.mode_parameters[2:] for waypoint in agent.mode_list] for agent in agents_list]
        waypointslists = [[waypoint.mode_parameters for waypoint in agent.mode_list] for agent in agents_list]
        legend_drone_rect = []
        legend_unsafe_set = []
        legend_text = []
        x_max = 0
        x_min = 10000
        y_max = 0
        y_min = 1000

        color = '#d9d9d9'
        if isinstance(unsafe_set, np.ndarray):
            unsafe_set = pc.box2poly(unsafe_set.T)
        if isinstance(unsafe_set, pc.Polytope):
            unsafe_set = [unsafe_set]
        for i in range(len(unsafe_set)):
            reg = unsafe_set[i].unsafe_set
            if type(reg) == pc.Polytope:
                reg = pc.Region(list_poly=[reg])
            for poly in reg.list_poly:
                poly = pc.projection(poly, [1, 2])
                # hull = ConvexHull(points)
                # poly_patch = Polygon(points, alpha=.5, color=color, fill=True)
                if poly.A.size != 0:
                    vert = ppm.duality.compute_polytope_vertices(poly.A, poly.b)
                    ppm.polygon.plot_polygon(vert, color=color, alpha = 1)
                    vert = np.array(vert)
                    x = vert[:, 0]
                    x_max = max(x_max, np.max(x))
                    x_min = min(x_min, np.min(x))
                    y = vert[:, 1]
                    y_max = max(y_max, np.max(y))
                    y_min = min(y_min, np.min(y))
        poly_patch = Polygon(vert, alpha=1, color=color, fill=True)
        legend_unsafe_set.append(poly_patch)
        legend_text.append(f"unsafe")

        for d in range(drone_num):
            curr_agent = agents_list[d]
            wp = waypointslists[d]

            # for mode in curr_agent.abs_mode_list:
            #     vir_unsafe = mode.unsafeset_list

            if len(wp[0]) == 4:
                if d == 0:
                    for i in range(0,len(wp)):
                        if i == 0:
                            plt.plot([wp[i][0],wp[i][2]], [wp[i][1],wp[i][3]], 'k', label='waypoints', linewidth= 0.3, markersize=0.1)
                            # plt.plot([wp[i][0],wp[i][2]], [wp[i][1],wp[i][3]], 'k', linewidth=0.3, markersize=0.1)
                            tmp = mlines.Line2D([], [], color='k', linewidth= 0.3, markersize=0.1)
                            legend_unsafe_set.append(tmp)
                            legend_text.append("segments")
                        else:
                            plt.plot([wp[i][0], wp[i][2]], [wp[i][1], wp[i][3]], 'k', linewidth=0.3,
                                     markersize=0.1)
                            # plt.plot([wp[i][0], wp[i][2]], [wp[i][1], wp[i][3]], 'k', linewidth=0.3, markersize=0.1)

                else:
                    for i in range(0,len(wp)):
                        plt.plot([wp[i][0],wp[i][2]], [wp[i][1],wp[i][3]], 'k', linewidth=0.3, markersize=0.1)
                        plt.plot([wp[i][0],wp[i][2]], [wp[i][1],wp[i][3]], 'k', linewidth=0.3, markersize=0.1)
            elif len(wp[0]) == 6:
                if d == 0:
                    for i in range(0,len(wp)):
                        if i == 0:
                            plt.plot([wp[i][0],wp[i][3]], [wp[i][1],wp[i][4]], 'k', label='waypoints', linewidth= 0.3, markersize=0.1)
                            plt.plot([wp[i][0],wp[i][3]], [wp[i][1],wp[i][4]], 'k', linewidth=0.3, markersize=0.1)
                        else:
                            plt.plot([wp[i][0], wp[i][3]], [wp[i][1], wp[i][4]], 'k', linewidth=0.3,
                                     markersize=0.1)
                            plt.plot([wp[i][0], wp[i][3]], [wp[i][1], wp[i][4]], 'k', linewidth=0.3, markersize=0.1)
                else:
                    for i in range(0,len(wp)):
                        plt.plot([wp[i][0],wp[i][3]], [wp[i][1],wp[i][4]], 'k', linewidth=0.3, markersize=0.1)
                        plt.plot([wp[i][0],wp[i][3]], [wp[i][1],wp[i][4]], 'k', linewidth=0.3, markersize=0.1)

            edge_list = curr_agent.edge_list
            # for i in range(len(edge_list)):
            #     guard = edge_list[i].guard
            #     if i == 0:
            #         plt.plot([guard[0,0], guard[1,0], guard[1,0], guard[0,0], guard[0,0]],\
            #                  [guard[0,1], guard[0,1], guard[1,1], guard[1,1], guard[0,1]], label='guard', color = 'k')
            #         tmp = mlines.Line2D([],[],color = 'k')
            #         legend_unsafe_set.append(tmp)
            #         legend_text.append("guards")
            #     else:
            #         plt.plot([guard[0, 0], guard[1, 0], guard[1, 0], guard[0, 0], guard[0, 0]], \
            #                  [guard[0, 1], guard[0, 1], guard[1, 1], guard[1, 1], guard[0, 1]], label='guard',
            #                  color='k')

            curr_tubes = tubes[d]
            mode_list = []
            for ci in range(len(curr_tubes)):
                curr_segment = curr_tubes[ci]
                for curr_tube in curr_segment.tube_list:
                    if sym_level == 2:
                        color = Plotter.colors[curr_segment.virtual_mode % len(Plotter.colors)]
                    else:
                        color = Plotter.colors[0]
                    # curr_tube = curr_segment.tube
                    curr_segments = curr_segment.trace
                    for i in range(0, len(curr_tube), plot_step):
                        # box_of_poly = PolyUtils.get_bounding_box(curr_tube[i])
                        # points = pc.extreme(curr_tube[i])
                        # points = points[:, :2]
                        # points = np.unique(points,axis = 0)
                        # poly = pc.Polytope(points)
                        # print(i)
                        poly = pc.projection(curr_tube[i],[1,2])
                        # hull = ConvexHull(points)
                        # poly_patch = Polygon(points, alpha=.5, color=color, fill=True)
                        if poly.A.size != 0:
                            vert = ppm.duality.compute_polytope_vertices(poly.A, poly.b)
                            ppm.polygon.plot_polygon(vert, color=color)
                            if i == 0 and curr_segment.virtual_mode not in mode_list:
                                poly_patch = Polygon(vert, alpha=.5, color=color, fill=True)
                                legend_unsafe_set.append(poly_patch)
                                legend_text.append(f"abs mode {curr_segment.virtual_mode}")
                                mode_list.append(curr_segment.virtual_mode)

                            vert = np.array(vert)
                            x = vert[:,0]
                            x_max = max(x_max, np.max(x))
                            x_min = min(x_min, np.min(x))
                            y = vert[:,1]
                            y_max = max(y_max, np.max(y))
                            y_min = min(y_min, np.min(y))
                        # rect = Rectangle(box_of_poly[0, [0, 1]], box_of_poly[1, 0] - box_of_poly[0, 0],
                        #                  box_of_poly[1, 1] - box_of_poly[0, 1], linewidth=1,
                        #                  edgecolor=color, facecolor=color)
                        # if ci == 0 and i == 0:
                            # legend_drone_rect.append(poly_patch)
                        # currentAxis.add_patch(poly_patch)

                    # for segment in curr_segments:
                    #     plt.plot(segment[:,0],segment[:,1],'b')
        for run in traces[d]:
            for segment in run:
                plt.plot(segment[:,0], segment[:,1],'b')
        color = '#80b1d3'
        for i in range(len(init_set)):
            reg = init_set[i]
            if type(reg) == pc.Polytope:
                reg = pc.Region(list_poly=[reg])
            for poly in reg.list_poly:
                points = pc.extreme(poly)
                if points is None:
                    continue
                points = points[:, :2]
                hull = ConvexHull(points)
                poly_patch = Polygon(points[hull.vertices, :], alpha=.5, color=color, fill=True)
                if i == 0:
                    legend_unsafe_set.append(poly_patch)
                    legend_text.append(f"init set")
                currentAxis.add_patch(poly_patch)

        color = '#8dd3c7'
        for i in range(len(goal_set)):
            reg = goal_set[i]
            if type(reg) == pc.Polytope:
                reg = pc.Region(list_poly=[reg])
            for poly in reg.list_poly:
                points = pc.extreme(poly)
                points = points[:, :2]
                hull = ConvexHull(points)
                poly_patch = Polygon(points[hull.vertices, :], alpha=.5, color=color, fill=True, label = "aa")
                if i == 0:
                    legend_unsafe_set.append(poly_patch)
                    legend_text.append(f"goal set")
                currentAxis.add_patch(poly_patch)

        # plt.legend(handles = legend_unsafe_set, labels = legend_text)
        # plt.ylim([-40, 40])
        # plt.xlim([-55, 55])
        #plt.ylim([-8, 26])
        #plt.xlim([-20, 75])
        #plt.ylim([-30, 30])
        #plt.xlim([-55, 40])
        #plt.ylim([-8, 26])
        #plt.xlim([-20, 75])
        #plt.ylim([-10, 30])
        #plt.xlim([-10, 60])
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlabel('x', fontsize=15)
        plt.ylabel('y', fontsize=15)
        plt.xlim([x_min-10,x_max+10])
        plt.ylim([y_min-10,y_max+10])
        # plt.legend(["aaaa"], handles = [poly_patch])
        #plt.tight_layout()
        # plt
        plt.show()

    # def plot_virtual(tubes: List[List[ReachtubeSegment]], agents_list: List[Agent], is_unified: bool, unsafe_set,
    #         plot_step=1):
    @staticmethod
    def plot_virtual(tube_dict: Dict[Tuple[float, ...], List[np.array]], agents_list: List[Agent], abs_nodes_neighbors,
                     plot_step=1):
        # agents_tubes = [[segment.tube for segment in segment_list] for segment_list in tubes]
        plt.figure()
        currentAxis = plt.gca()
        # waypointslists = [[waypoint.mode_parameters for waypoint in agent.path] for agent in agents_list]
        legend_drone_rect = []
        legend_unsafe_set = []
        legend_text = []

        color_indx = 0

        cur_agent = agents_list[-1]
        abs_edge_list = cur_agent.abs_edge_list

        # for abs_edge in abs_edge_list:
        #     guard = pc.bounding_box(abs_edge.region_guard)
        #     plt.plot([guard[0][0], guard[1][0], guard[1][0], guard[0][0], guard[0][0]], \
        #          [guard[0][1], guard[0][1], guard[1][1], guard[1][1], guard[0][1]], color='k')
        tmp = mlines.Line2D([], [], color='k')
        legend_unsafe_set.append(tmp)
        legend_text.append("guard")

        tmp = mlines.Line2D([], [], color='k', linewidth=0.1, markersize=1)
        legend_unsafe_set.append(tmp)
        legend_text.append("segments")

        for virtual_mode in tube_dict:
            # agents_list[-1].mode_to_abs_mode[virtual_mode]
            color = Plotter.colors[virtual_mode % len(Plotter.colors)]
            curr_tube_list = tube_dict[virtual_mode]
            # print("tube of mode ", virtual_mode, " is:", curr_tube)
            '''
            for edge_id in abs_nodes_neighbors[virtual_mode]:
                next_node = agents_list[-1].abs_edge_list[edge_id].dest
                before_mode = agents_list[-1].abs_mode_list[virtual_mode].mode_parameters[2:]
                after_mode = agents_list[-1].abs_mode_list[next_node].mode_parameters[2:]
                print(virtual_mode, "->", next_node)
                plt.plot([before_mode[0], after_mode[0]], [before_mode[1], after_mode[1]], 'k' + 'o', label='waypoints', linewidth=1, markersize=1)
                plt.plot([before_mode[0], after_mode[0]], [before_mode[1], after_mode[1]], 'k', linewidth=1, markersize=1)
            '''
            wp = agents_list[-1].abs_mode_list[virtual_mode].mode_parameters
            if len(wp) == 4:
                plt.plot([wp[0],wp[2]], [wp[1],wp[3]], 'ko', label='waypoints', linewidth= 1, markersize=2)
                plt.plot([wp[0],wp[2]], [wp[1],wp[3]], 'k', linewidth=1, markersize=2)

            elif len(wp) == 6:
                plt.plot([wp[0],wp[3]], [wp[1],wp[4]], 'ko', label='waypoints', linewidth= 1, markersize=2)
                plt.plot([wp[0],wp[3]], [wp[1],wp[4]], 'k', linewidth=1, markersize=2)

                tmp = mlines.Line2D([], [], color='k', linewidth=1, markersize=2)
                legend_unsafe_set.append(tmp)
                legend_text.append("Reference")
            """
            wp = np.array(waypointslists[d])
            if d == 0:
                plt.plot(wp[:, 0], wp[:, 1], 'k' + 'o', label='waypoints')
                plt.plot(wp[:, 0], wp[:, 1], 'k')
            else:
                plt.plot(wp[:, 0], wp[:, 1], 'k' + 'o')
                plt.plot(wp[:, 0], wp[:, 1], 'k')
            """
            for curr_tube in curr_tube_list:
                for i in range(0, len(curr_tube), plot_step):
                    box_of_poly = curr_tube[i]  # PolyUtils.get_region_bounding_box(curr_tube[i])
                    rect = Rectangle(box_of_poly[0, [0, 1]], box_of_poly[1, 0] - box_of_poly[0, 0],
                                     box_of_poly[1, 1] - box_of_poly[0, 1], linewidth=1,
                                     edgecolor=color, facecolor=color)
                    if i == 0:
                        legend_unsafe_set.append(rect)
                        legend_text.append(f"abs mode {virtual_mode}")
                    currentAxis.add_patch(rect)
            color = '#d9d9d9'
            unsafeset_list = agents_list[-1].abs_mode_list[virtual_mode].unsafeset_list
            for j in range(len(unsafeset_list)):
                unsafe_set = unsafeset_list[j].unsafe_set
                if isinstance(unsafe_set, np.ndarray):
                    unsafe_set = pc.box2poly(unsafe_set.T)
                if isinstance(unsafe_set, pc.Polytope):
                    unsafe_set = [unsafe_set]
                for i in range(len(unsafe_set)):
                    poly = unsafe_set[i]
                    points = pc.extreme(poly)
                    points = points[:, :2]
                    hull = ConvexHull(points)
                    poly_patch = Polygon(points[hull.vertices, :], alpha=1, color=color, fill=True)
                    if i == 0 and j == 0:
                        legend_unsafe_set.append(poly_patch)
                        legend_text.append(f"abs unsafe {virtual_mode}")
            
                    # currentAxis.add_patch(poly_patch)
            color_indx = color_indx + 1


        # plt.legend(handles = legend_unsafe_set, labels = legend_text)
        # plt.ylim([-30, 30])
        # plt.xlim([-30, 30])
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlabel('x', fontsize=15)
        plt.ylabel('y', fontsize=15)
        plt.tight_layout()
        # plt.legend()
        # plt
        plt.show()