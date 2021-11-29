from typing import List
import math
import numpy as np


class PolyUtils:

    @staticmethod
    def does_rect_contain(rect1, rect2): # does rect2 contains rect1
        for i in range(len(rect1[0][:])):
            if rect1[0][i] < rect2[0][i] or rect1[1][i] > rect2[1][i]:
                return False
        return True

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
                return False
        return True

    @staticmethod
    def get_rects_inter(rect1: np.array, rect2: np.array):
        result = np.empty(rect1.shape)
        result[0, :] = np.maximum(rect1[0, :], rect2[0, :])
        result[1, :] = np.minimum(rect1[1, :], rect2[1, :])
        # print("The result of the intersection is:", result)
        return result

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


if __name__ == '__main__':
    print("polyutils file")


