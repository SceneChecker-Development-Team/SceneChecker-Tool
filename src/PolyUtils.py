import polytope as pc
import numpy as np
from typing import List, Dict
import pdb
from scipy.spatial import ConvexHull
import math
import dill
import ray
import picos as pic
import cvxopt as cvx
import numpy as np


class PolyUtils:

    @staticmethod
    def ceil(val, precision = 0):
        factor = int(math.pow(10,precision))
        return math.ceil(val*factor)/factor

    @staticmethod
    def floor(val, precision = 0):
        factor = int(math.pow(10, precision))
        return math.floor(val * factor) / factor

    @staticmethod
    def does_rect_contain(rect1, rect2): # does rect2 contains rect1
        for i in range(len(rect1[0][:])):
            if PolyUtils.ceil(rect1[0][i],0) < PolyUtils.floor(rect2[0][i],0) or PolyUtils.floor(rect1[1][i],0) > PolyUtils.ceil(rect2[1][i],0):
            # if math.ceil(rect1[0][i]) < math.floor(rect2[0][i]) or math.floor(rect1[1][i]) > math.ceil(rect2[1][i]):
                #print("containement: ", math.ceil(rect1[0][i]), "non rounded:", rect1[0][i], "rounded: ",  math.floor(rect2[0][i]), "non rounded: ", rect2[0][i])
                #print("containement: ", math.floor(rect1[1][i]), "non rounded:", rect1[1][i], "rounded: ", math.ceil(rect2[1][i]), "non rounded: ", rect2[1][i])
                return False
        return True

    @staticmethod
    def get_rect_volume(rect1):  # does rect2 contains rect1
        result = 1
        for i in range(len(rect1[0][:])):
            result = result * (rect1[1][i]  - rect1[0][i])
        return result

    @staticmethod
    def is_Hrep_empty(A, b):
        prob = pic.Problem(verbose = -1);
        x = prob.add_variable('x', A.shape[1]);
        prob.add_constraint(pic.new_param('A', cvx.matrix(A)) * x <= pic.new_param('b', cvx.matrix(b)));
        prob.set_objective('find', x[0])
        cvx.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        try:
            prob.solve(verbose=False, solver='cvxopt')
        except ValueError:
            print("Error in checking if poly is empty")
            print("A", A, "b", b)
            return True
        # print(prob.status)
        if prob.status == 'optimal':
            return False
        else:
            return True

    @staticmethod
    def is_polytope_intersection_empty(poly1: pc.Polytope, poly2: pc.Polytope) -> bool:
        if pc.is_empty(poly1) or pc.is_empty(poly2):
            raise ValueError("checking intersection between empty polytopes")
        return pc.is_empty(pc.intersect(poly1, poly2))
        # return PolyUtils.is_Hrep_empty(np.row_stack((poly1.A, poly2.A)), np.concatenate((poly1.b, poly2.b)))


    @staticmethod
    def check_rect_empty(rect: np.array) -> bool:
        return not np.all(rect[0, :] <= rect[1, :])

    @staticmethod
    def do_rects_inter(rect1: np.array, rect2: np.array):
        #print("rects 1 and 2:", rect1, rect2)
        #print(rect1.shape, rect2.shape)
        if PolyUtils.check_rect_empty(rect1) or PolyUtils.check_rect_empty(rect2):
            print("rects:", rect1, rect2)
            raise ValueError("Do no pass empty rectangles to intersect function")
        for i in range(rect1.shape[1]):
            if rect1[0, i] > rect2[1, i] or rect1[1, i] < rect2[0, i]:
                # print("i: ", i)
                return False
        #print("they do intersect")
        return True

    @staticmethod
    def get_rects_inter(rect1: np.array, rect2: np.array):
        result = np.empty(rect1.shape)
        result[0, :] = np.maximum(rect1[0, :], rect2[0, :])
        result[1, :] = np.minimum(rect1[1, :], rect2[1, :])
        # print("The result of the intersection is:", result)
        return result

    @staticmethod
    def get_region_bounding_box(reg: pc.Region) -> np.array:
        if type(reg) == pc.Polytope:
            print("warning, called region bbox function on polytope")
            return PolyUtils.get_bounding_box(reg)
        elif len(reg.list_poly) <= 0 or pc.is_empty(reg):
            raise ValueError("Passed an empty region, no valid bbox")
        return np.row_stack((np.min(np.stack(map(PolyUtils.get_bounding_box, reg.list_poly))[:, 0, :], axis=0),
                             np.max(np.stack(map(PolyUtils.get_bounding_box, reg.list_poly))[:, 1, :], axis=0)))

    @staticmethod
    def get_bounding_box(poly: pc.Polytope, verbose=False) -> np.array:
        if type(poly) == pc.Region:
            return PolyUtils.get_region_bounding_box(poly)
        elif type(poly) != pc.Polytope:
            # print(type(poly))
            raise TypeError("this function only takes polytopes")
        poly.bbox = None
        if verbose:
            print("working")
        return np.column_stack(poly.bounding_box).T


    """
    @staticmethod
    def get_convex_union(list_poly: List[pc.Polytope]) -> pc.Polytope:
        if any(type(poly) != pc.Polytope for poly in list_poly):
            raise TypeError("Only accepts list of polytopes")
        try:
            all_points = np.row_stack(map(pc.extreme, list_poly))
        except ValueError:
            pdb.set_trace()
        return pc.reduce(pc.qhull(all_points))
    """
    @staticmethod
    def get_convex_union(list_array: List[np.array]) -> np.array:
        """
        if any(type(rect) != np.array for rect in list_array):
            print([type(rect) != np.array for rect in list_array])
            print("list array:", list_array)
            raise TypeError("Only accepts list of arrays")
        try:
        """
        assert len(list_array) > 0, "list array length should be larger than zero"
        result: np.array = np.copy(list_array[0])
        for i in range(1, len(list_array)):
            result[0, :] = np.minimum(result[0, :], list_array[i][0, :])
            result[1, :] = np.maximum(result[1, :], list_array[i][1, :])
        return result

    @staticmethod
    def get_region_union(old_reg: pc.Region, new_reg: pc.Region) -> np.array:
        assert (type(old_reg) == pc.Region or type(old_reg) == pc.Polytope) \
               and (type(new_reg) == pc.Region or type(new_reg) == pc.Polytope), "arguments should be regions or polytopes "

        if type(new_reg) == pc.Polytope:
            new_reg = pc.Region(list_poly=[new_reg])
        for new_ind, new_poly in enumerate(new_reg.list_poly):
            existing_poly = False
            new_rect = PolyUtils.get_bounding_box(new_poly)
            if type(old_reg) == pc.Polytope:
                old_reg = pc.Region(list_poly=[old_reg])
            for ind, poly in enumerate(old_reg.list_poly):
                old_rect = PolyUtils.get_bounding_box(poly)
                if PolyUtils.do_rects_inter(old_rect, new_rect): # PolyUtils.does_rect_contain(new_rect, old_rect):  # PolyUtils.do_rects_inter(old_rect, new_rect): #
                    #unioned_rect = PolyUtils.get_convex_union([old_rect, new_rect])
                    #old_reg.list_poly[ind] = pc.box2poly(unioned_rect.T)
                    existing_poly = True
                    break
            if not existing_poly:
                old_reg = \
                    pc.union(new_poly, old_reg)
        return old_reg

    @staticmethod
    def subtract_regions(new_reg: pc.Region, old_reg: pc.Region) -> np.array: # subtract the old from the new
        assert (type(old_reg) == pc.Region or type(old_reg) == pc.Polytope) \
               and (type(new_reg) == pc.Region or type(new_reg) == pc.Polytope), "arguments should be regions"
        result = pc.Region(list_poly=[])
        if not pc.is_subset(new_reg, old_reg):
            if type(new_reg) == pc.Polytope:
                new_reg = pc.Region(list_poly=[new_reg])
            if type(old_reg) == pc.Polytope:
                old_reg = pc.Region(list_poly=[old_reg])
            for new_poly in new_reg.list_poly:
                new_rect = PolyUtils.get_bounding_box(new_poly)
                rect_contain = False
                for old_poly in old_reg.list_poly:
                    old_rect = PolyUtils.get_bounding_box(old_poly)
                    if PolyUtils.does_rect_contain(new_rect, old_rect):
                        rect_contain = True
                        break
                if not rect_contain:
                    result = PolyUtils.get_region_union(result, new_poly)
        return result

    # TODO confirm deletion of this
    # region center may not be inside region.
    """
    @staticmethod
    def get_region_center(reg: pc.Region) -> np.array:
        bbox = PolyUtils.get_region_bounding_box(reg)
        center = np.array([])
        for i in range(bbox[0, :].shape[0]):
            center.append((bbox[0, i] + bbox[1, i]) / 2)
    """

    # TODO move this to transform abstraction
    @staticmethod
    def point_coordinate_change(p: np.array, rot_angle: float, translation_vector: np.array):
        print("Input: ", p, rot_angle, translation_vector, sc)
        x1 = p[0]  # x_i
        y1 = p[1]  # y_i
        s1 = p[2]  # psi
        v1 = p[3]  # velocity
        x2 = (x1 - translation_vector[0]) * math.cos(rot_angle) + (y1 - translation_vector[1]) * math.sin(rot_angle)
        y2 = -1 * (x1 - translation_vector[0]) * math.sin(rot_angle) + (y1 - translation_vector[1]) * math.cos(rot_angle)
        s2 = s1 + rot_angle
        v2 = v1
        print("Output: ", np.array([x2, y2, s2, v2]))
        return np.array([x2, y2, s2, v2])


    @staticmethod
    def print_region(reg: pc.Region):
        for poly in reg.list_poly:
            print(poly)



    @staticmethod
    def project_to_intersect(polyreg1: pc.Region, polyreg2: pc.Polytope):
        if isinstance(polyreg1, pc.Polytope) and isinstance(polyreg2, pc.Polytope):
            A: np.array = np.zeros((polyreg1.A.shape[0], polyreg2.dim))
            A[:, :polyreg1.A.shape[1]] = polyreg1.A
            return pc.Polytope(A=A, b=polyreg1.b)
        elif isinstance(polyreg1, pc.Polytope):
            res = []
            for poly in polyreg2.list_poly:
                res.append(PolyUtils.project_to_intersect(polyreg1, poly))
            return pc.Region(list_poly=res)
        elif isinstance(polyreg2, pc.Polytope):
            res = []
            for poly in polyreg1.list_poly:
                res.append(PolyUtils.project_to_intersect(poly, polyreg2))
            return pc.Region(list_poly=res)
        else:
            res = []
            for poly1 in polyreg1.list_poly:
                for poly2 in polyreg2.list_poly:
                    res.append(PolyUtils.project_to_intersect(poly1, poly2))
            return pc.Region(list_poly=res)

    """
    @staticmethod
    @ray.remote
    def merger(i: int, big_tube):
        cur_region: List[pc.Polytope] = []
        for ind in range(len(big_tube)):
            if len(big_tube[ind]) > i:
                cur_region.append(big_tube[ind][i])
        if len(cur_region) == 1:
            return cur_region[0]
        elif len(cur_region) > 1:
            return PolyUtils.get_convex_union(cur_region)
        else:
         raise ValueError("Unable to merge any tubes at index", i)
    """
    """
    @staticmethod
    @ray.remote
    def merger(i: int, big_tube):
        cur_tube: List[pc.array] = []
        for ind in range(len(big_tube)):
            if len(big_tube[ind]) > i:
                cur_tube.append(big_tube[ind][i])
        if len(cur_tube) == 1:
            return cur_tube[0]
        elif len(cur_tube) > 1:
            return PolyUtils.get_convex_union(cur_tube)
        else:
            raise ValueError("Unable to merge any tubes at index", i)

    @staticmethod
    def merge_tubes(big_tubes: List[List[np.array]]) -> List[np.array]:
        ray.init(ignore_reinit_error=True)
        max_ind = max(map(len, big_tubes))
        futures = [PolyUtils.merger.remote(i, big_tubes) for i in range(max_ind)]
        result = ray.get(futures)
        return result
    
    """

    @staticmethod
    def merge_tubes(big_tubes: List[List[np.array]]) -> List[np.array]:
        max_ind = max(map(len, big_tubes))
        result = []
        for i in range(max_ind):
            list_t = []
            for j in range(len(big_tubes)):
                if i >= len(big_tubes[j]):
                    continue
                list_t.append(big_tubes[j][i])
            result.append(PolyUtils.get_convex_union(list_t))
        return result

    # mutates big_tube
    """
    @staticmethod
    def merge_tubes(big_tubes: List[List[pc.Polytope]]) -> List[pc.Polytope]:
        ray.init(ignore_reinit_error=True)
        max_ind = max(map(len, big_tubes))
        futures = [PolyUtils.merger.remote(i, big_tubes) for i in range(max_ind)]
        result = ray.get(futures)
        return result
    """

    @staticmethod
    def correct_rows_order(poly: pc.Polytope):
        """
        :param poly: given a polytope from polytope library
        :return: returns a polytope from polytope library with same A matrix row ordering as from box2poly
        """
        if pc.is_empty(poly):
            pdb.set_trace()
        offset = 0
        # for i in range(poly.A.shape[0]):
        #    for j in range(poly.A.shape[1]):
        #        poly.A[i][j] = float(int(poly.A[i][j]))

        box = pc.bounding_box(poly)
        poly = pc.box2poly(np.column_stack((box[0], box[1])))

        # print "poly:", poly
        poly = pc.reduce(poly)
        # print "poly:", poly
        for row_idx_1 in range(poly.A.shape[0]):
            if row_idx_1 >= poly.A.shape[0] / 2:
                offset = int(poly.A.shape[0] / 2)
            if poly.A[row_idx_1][row_idx_1 - offset] == 0:
                for row_idx_2 in range(poly.A.shape[0]):
                    if (offset == 0 and poly.A[row_idx_2][row_idx_1 - offset] == 1) or (
                            offset != 0 and poly.A[row_idx_2][row_idx_1 - offset] == -1):
                        poly.A[[row_idx_1, row_idx_2]] = poly.A[[row_idx_2, row_idx_1]]
                        poly.b[[row_idx_1, row_idx_2]] = poly.b[[row_idx_2, row_idx_1]]
                        break
        return poly

    @staticmethod
    def rect_poly_project(poly: pc.Polytope, dim: np.array):
        """
        :param poly: reduces dimensionality of polytope according to dim param
        :param dim: specifies constraints to remove by dimension index
        :param poly_has_time: boolean for if the provided polytope has a time dimension
        :return:
        """
        # poly = PolyUtils.correct_rows_order(poly)
        A = poly.A
        b = poly.b
        n = A.shape[1]
        for i in range(A.shape[1]):
            if -1 * b[n + i] > b[i]:
                pdb.set_trace()
        dim_c = dim
        dim_c = np.ndarray.tolist(dim_c)
        A_new = A[:, dim_c]
        p = pc.Polytope(A_new, b)
        n = p.A.shape[1]
        for i in range(p.A.shape[1]):
            if -1 * p.b[n + i] > p.b[i]:
                pdb.set_trace()
        return p

    @staticmethod
    def region_projection(reg, dim):
        """
            :param poly: reduces dimensionality of polytope according to dim param
            :param dim: specifies constraints to remove by dimension index
            :param poly_has_time: boolean for if the provided polytope has a time dimension
            :return: reduced dimensionality polytope
        """
        proj_reg = pc.Region(list_poly=[])
        # print "list poly:", reg.list_poly
        # for i in range(len(reg.list_poly)):
        #   print reg.list_poly[i]
        for poly in reg.list_poly:
            # print "poly in region: ", poly
            if pc.is_empty(poly):
                pdb.set_trace()
            proj_reg = pc.union(proj_reg, PolyUtils.rect_poly_project(poly, dim))
            if pc.is_empty(proj_reg):
                print("poly_reg is empty")
                pdb.set_trace()
        return proj_reg

    @staticmethod
    def dist_to_polytope(box: np.array, point: np.array):
        if box.shape[1] != point.shape[0]:
            print(box.shape, point.shape)
            raise ValueError
        res = []
        for i in range(point.shape[0]):
            tmp = min(abs(point[i]-box[0][i]),abs(point[i]-box[1][i]))
            if (point[i]-box[0][i]) * (point[i]-box[1][i]) < 0:
                tmp = 0
            res.append(tmp)
        return np.linalg.norm(np.array(res))


if __name__ == '__main__':
    rect1 = np.array([[-1., -7., -1.,  0.], [0., -6.,  0., 1.]])
    rect2 = np.array([[-0.61899, -6.03102, -0.95163,  0.5 ], [0.61899, -4.79303, -0.94163,  0.6 ]])
    reg = pc.Region(list_poly=[pc.box2poly(np.array([[11.9, 5.1, -10, -100], [12.9, 6.1, 10, 100]]).T)])
    # PolyUtils.print_region(reg)
    sc = 1.32581766367
    translation_vector = np.array([-3.5, -0.5, -1 * sc, 0])
    print("bounding box: ", PolyUtils.get_region_bounding_box(reg))
    reg_transformed = PolyUtils.transform_poly_to_unified(reg, translation_vector, sc)
    # PolyUtils.print_region(reg_transformed)
    # points = pc.extreme(reg_transformed.list_poly[0])
    # hull = ConvexHull(points)
    # print("points:", points[hull.vertices, :2])
    print("bounding box: ", PolyUtils.get_region_bounding_box(reg_transformed))

    reg_transformed_back = PolyUtils.transform_region_from_unified(reg_transformed, -1 * translation_vector, -1 * sc)

    print("bounding box: ", PolyUtils.get_region_bounding_box(reg_transformed_back))


