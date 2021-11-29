# Plot polytope in 3d
# Written by: Kristina Miller

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3

from scipy.spatial import ConvexHull
import pypoman as ppm
from src.PolyUtils import PolyUtils

from src.ReachtubeSegment import ReachtubeSegment
from src.Agent import Agent
from typing import List
import polytope as pc
from typing import List, Dict, Optional, Tuple

class Faces():
    def __init__(self,tri, sig_dig=12, method="convexhull"):
        self.method=method
        self.tri = np.around(np.array(tri), sig_dig)
        self.grpinx = list(range(len(tri)))
        norms = np.around([self.norm(s) for s in self.tri], sig_dig)
        _, self.inv = np.unique(norms,return_inverse=True, axis=0)

    def norm(self,sq):
        cr = np.cross(sq[2]-sq[0],sq[1]-sq[0])
        return np.abs(cr/np.linalg.norm(cr))

    def isneighbor(self, tr1,tr2):
        a = np.concatenate((tr1,tr2), axis=0)
        return len(a) == len(np.unique(a, axis=0))+2

    def order(self, v):
        if len(v) <= 3:
            return v
        v = np.unique(v, axis=0)
        n = self.norm(v[:3])
        y = np.cross(n,v[1]-v[0])
        y = y/np.linalg.norm(y)
        c = np.dot(v, np.c_[v[1]-v[0],y])
        if self.method == "convexhull":
            h = ConvexHull(c)
            return v[h.vertices]
        else:
            mean = np.mean(c,axis=0)
            d = c-mean
            s = np.arctan2(d[:,0], d[:,1])
            return v[np.argsort(s)]

    def simplify(self):
        for i, tri1 in enumerate(self.tri):
            for j,tri2 in enumerate(self.tri):
                if j > i:
                    if self.isneighbor(tri1,tri2) and \
                       self.inv[i]==self.inv[j]:
                        self.grpinx[j] = self.grpinx[i]
        groups = []
        for i in np.unique(self.grpinx):
            u = self.tri[self.grpinx == i]
            u = np.concatenate([d for d in u])
            u = self.order(u)
            groups.append(u)
        return groups

class Plotter():
    colors = ['#b3de69',  '#bebada', '#fdb462', '#fccde5', '#bc80bd', '#ccebc5', '#ffed6f', '#ffffb3', '#c7eae5']
    
    @staticmethod
    def plot_polytope_3d(A, b, ax = None, edgecolor = 'k', color = 'red', trans = 0.2):
        verts = np.array(ppm.compute_polytope_vertices(A, b))
        # compute the triangles that make up the convex hull of the data points
        hull = ConvexHull(verts)
        triangles = [verts[s] for s in hull.simplices]
        # combine co-planar triangles into a single face
        faces = Faces(triangles, sig_dig=1).simplify()
        # plot
        if ax == None:
            ax = a3.Axes3D(plt.figure())

        pc = a3.art3d.Poly3DCollection(faces,
                                       facecolor=color,
                                       edgecolor=edgecolor, alpha=trans)
        ax.add_collection3d(pc)
        # define view
        yllim, ytlim = ax.get_ylim()
        xllim, xtlim = ax.get_xlim()
        zllim, ztlim = ax.get_zlim()
        x = verts[:,0]
        x = np.append(x, [xllim+1, xtlim-1])
        y = verts[:,1]
        y = np.append(y, [yllim+1, ytlim-1])
        z = verts[:,2]
        z = np.append(z, [zllim+1, ztlim-1])
        # print(np.min(x)-1, np.max(x)+1, np.min(y)-1, np.max(y)+1, np.min(z)-1, np.max(z)+1)
        ax.set_xlim(np.min(x)-1, np.max(x)+1)
        ax.set_ylim(np.min(y)-1, np.max(y)+1)
        ax.set_zlim(np.min(z)-1, np.max(z)+1)
        # ax.set_xlim(-1, 19)
        # ax.set_ylim(-1, 19)
        # ax.set_zlim(-1, 19)

    @staticmethod
    def plot_line_3d(start, end, ax = None, color = 'blue'):
        x = [start[0], end[0]]
        y = [start[1], end[1]]
        z = [start[2], end[2]]
        line = a3.art3d.Line3D(x,y,z,color = color)
        ax.add_line(line)

    @staticmethod
    def plot(drone_num, tubes: List[List[ReachtubeSegment]], agent_trace, agents_list: List[Agent], unsafe_set,
             init_set = [], goal_set = [], sym_level = 0, plot_step=1):
        ax1 = a3.Axes3D(plt.figure())
        ax1.set_xlim(0,1)
        ax1.set_ylim(0,1)
        ax1.set_zlim(0,1)
        waypointlists = [[waypoint.mode_parameters for waypoint in agent.mode_list] for agent in agents_list]
        
        color = '#d9d9d9'
        for i in range(len(unsafe_set)):
            unsafe = unsafe_set[i].unsafe_set
            if type(unsafe) == pc.Polytope:
                unsafe = pc.Region(list_poly=[unsafe])
            for poly in unsafe.list_poly:
                poly = pc.projection(poly, [1,2,3])
                Plotter.plot_polytope_3d(poly.A,poly.b,ax = ax1, edgecolor='k', color = color, trans=0.5)
        
        for d in range(drone_num):
            curr_agent = agents_list[d]
            wp = waypointlists[d]
            for i in range(1,len(wp)):
                Plotter.plot_line_3d([wp[i][0],wp[i][1],
                    wp[i][2]],[wp[i][3],wp[i][4],wp[i][5]], ax = ax1, color = 'k')
            
            curr_tubes = tubes[d]
            for ci in range(0, len(curr_tubes), plot_step):
                curr_segment = curr_tubes[ci]
                curr_tube_list = curr_segment.tube_list
                for curr_tube in curr_tube_list:
                    if sym_level == 2:
                        color = Plotter.colors[curr_segment.virtual_mode % len(Plotter.colors)]
                    else:
                        color = Plotter.colors[0]     
                    for i in range(0, len(curr_tube), plot_step):
                        box_of_poly = PolyUtils.get_bounding_box(curr_tube[i])
                        A,b = Plotter.convert_interval_to_matrix(box_of_poly)
                        Plotter.plot_polytope_3d(A, b, ax = ax1, edgecolor='k',color = color, trans=1)

        color = '#80b1d3'
        for i in range(len(init_set)):
            reg = init_set[i]
            if type(reg) == pc.Polytope:
                reg = pc.Region(list_poly = [reg])
            for poly in reg.list_poly:
                poly = pc.projection(poly, [1, 2, 3])
                Plotter.plot_polytope_3d(poly.A,poly.b,ax = ax1, edgecolor='k', color = color, trans=0.5)

        color = '#8dd3c7'
        for i in range(len(goal_set)):
            reg = goal_set[i]
            if type(reg) == pc.Polytope:
                reg = pc.Region(list_poly = [reg])
            for poly in reg.list_poly:
                poly = pc.projection(poly, [1, 2, 3])
                Plotter.plot_polytope_3d(poly.A,poly.b,ax = ax1, edgecolor='k', color = color, trans=0.5)

        plt.show()

    @staticmethod   
    def plot_virtual(tube_dict: Dict[Tuple[float, ...], List[np.array]], agents_list: List[Agent], abs_nodes_neighbors,
                     plot_step=1):
        ax1 = a3.Axes3D(plt.figure())
        ax1.set_xlim(0,1)
        ax1.set_ylim(0,1)
        ax1.set_zlim(0,1)

        color_index = 0
        for virtual_mode in tube_dict:
            color = Plotter.colors[virtual_mode % len(Plotter.colors)]
            curr_tube_list = tube_dict[virtual_mode]
            wp = agents_list[-1].abs_mode_list[virtual_mode].mode_parameters
            
            Plotter.plot_line_3d([wp[0],wp[1],wp[2]],[wp[3],wp[4],wp[5]], ax = ax1, color = 'k')
            
            for curr_tube in curr_tube_list:
                for i in range(0, len(curr_tube), plot_step):
                    box_of_poly = curr_tube[i]  # PolyUtils.get_region_bounding_box(curr_tube[i])
                    poly = pc.box2poly(box_of_poly.T)
                    poly = pc.projection(poly,[1,2,3])
                    Plotter.plot_polytope_3d(poly.A,poly.b,ax = ax1, edgecolor='k', color = color, trans=0.5)
            color_index = color_index + 1


        plt.show()

    @staticmethod
    def convert_interval_to_matrix(interval):
        A = np.array([[-1,0,0],
                      [1,0,0],
                      [0,-1,0],
                      [0,1,0],
                      [0,0,-1],
                      [0,0,1]])
        b = np.array([[-interval[0][0]],[interval[1][0]],[-interval[0][1]],[interval[1][1]],[-interval[0][2]],[interval[1][2]]])
        return A,b

if __name__ == '__main__':
    A = np.array([[-1, 0, 0],
                  [1, 0, 0],
                  [0, -1, 0],
                  [0, 1, 0],
                  [0, 0, -1],
                  [0, 0, 1]])
    b = np.array([[1], [1], [1], [1], [1], [1]])
    b2 = np.array([[-1], [2], [-1], [2], [-1], [2]])
    ax1 = a3.Axes3D(plt.figure())
    Plotter.Plplot_polytope_3d(A, b, ax = ax1, color = 'red')
    Plotter.plot_polytope_3d(A, b2, ax = ax1, color = 'green')
    plt.show()
