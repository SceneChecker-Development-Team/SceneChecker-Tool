import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def plot_rtsegment_and_traces(rtsegment: np.ndarray, traces: np.ndarray):
    for dim_ind in range(1, traces.shape[2]):
        fig, ax = plt.subplots(1)
        facecolor = 'r'
        for trace_ind in range(traces.shape[0]):
            ax.plot(traces[trace_ind, :, 0], traces[trace_ind, :, dim_ind])
        for hrect_ind in range(rtsegment.shape[0]):
            ax.add_patch(Rectangle((rtsegment[hrect_ind, 0, 0], rtsegment[hrect_ind, 0, dim_ind]), rtsegment[hrect_ind, 1, 0]-rtsegment[hrect_ind, 0, 0],
                                            rtsegment[hrect_ind, 1, dim_ind] - rtsegment[hrect_ind, 0, dim_ind], alpha=0.1, facecolor='r'))
        ax.set_title(f'dim #{dim_ind}')
        fig.canvas.draw()
        plt.show()
