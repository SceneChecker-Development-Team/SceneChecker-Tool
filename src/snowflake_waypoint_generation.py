import math as m
import numpy as np
import pdb
import matplotlib.pyplot as plt


def generate_snowflake_waypoints(side_len = 1, n=1):
    P = np.array([np.array([0, 0]), np.array([side_len, 0]), np.array([side_len * m.cos(-m.pi / 3), side_len * m.sin(-m.pi / 3)]), np.array([0, 0])])
    for i in range(n):
        newP = np.zeros((P.shape[0] * 4 + 1, 2))
        for j in range(P.shape[0]-1):
            newP[4 * j + 1, :] = P[j, :]
            print("newp: ", newP[4 * j + 1, :], "P: ", P[j, :])
            newP[4 * j + 2, :] = (2 * P[j, :] + P[j + 1, :]) / 3
            link = P[j + 1, :] - P[j, :]
            ang = m.atan2(link[1], link[0])
            linkLeng = m.sqrt(link[0] * link[0] + link[1] * link[1])
            newP[4 * j + 3, :] = newP[4 * j + 2, :] + (linkLeng / 3) * np.array(
                [m.cos(ang + m.pi / 3), m.sin(ang + m.pi / 3)])
            newP[4 * j + 4, :] = (P[j, :] + 2 * P[j + 1, :]) / 3

        newP[4 * P.shape[0] , :] = P[P.shape[0]-1, :]
        P = newP

    return P


def main():
    P = generate_snowflake_waypoints(100, 2)
    plt.plot(P[:,0], P[:,1])
    plt.show()
    pdb.set_trace()


if __name__ == '__main__':
    main()
