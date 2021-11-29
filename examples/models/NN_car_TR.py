import torch
from scipy.integrate import ode
import numpy as np
import polytope as pc
from typing import Optional, List, Tuple
import math
try:
    from src.Waypoint import Waypoint
except:
    from Waypoint import Waypoint
import matplotlib.pyplot as plt
from src.PolyUtils import PolyUtils

class FFNNC(torch.nn.Module):
    def __init__(self, D_in=4, H1=100):
        super(FFNNC, self).__init__()
        self.control1 = torch.nn.Linear(D_in, H1)
        self.control2 = torch.nn.Linear(H1, 2)

    def forward(self, x):
        h2 = torch.relu(self.control1(x))
        u = self.control2(h2)
        return u


def vehicle_dynamics(t, vars, args):
    curr_x = vars[3]
    curr_y = vars[4]
    curr_theta = vars[5] % (np.pi * 2)
    vr = args[0]
    delta = args[1]
    bx = args[2]
    by = args[3]

    if vr > 100:
        vr = 100
    elif vr < -0:
        vr = -0

    if delta > np.pi / 3:
        delta = np.pi / 3
    elif delta < -np.pi / 3:
        delta = -np.pi / 3

    # beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
    # dx = vr*np.cos(curr_theta+beta)
    # dy = vr*np.sin(curr_theta+beta)
    # dtheta = vr/Lr * np.sin(beta)
    dx = vr * np.cos(curr_theta + delta)
    dy = vr * np.sin(curr_theta + delta)
    dtheta = delta
    dref_x = bx
    dref_y = by
    dref_theta = 0
    return [dref_x, dref_y, dref_theta, dx, dy, dtheta]


def runModel(initalCondition, time_bound, time_step, ref_input):
    controller = FFNNC()
    controller.load_state_dict(torch.load('./model_controller'))

    init = initalCondition
    trajectory = [init]
    r = ode(vehicle_dynamics)
    r.set_initial_value(init)
    t = 0
    time = [t]
    trace = [[t]]
    trace[0].extend(init[3:])
    i = 0
    sc = ref_input[2]
    theta_transform = ref_input[2] - np.pi / 2
    while t <= time_bound:
        error_x = trajectory[i][0] - trajectory[i][3]
        error_y = trajectory[i][1] - trajectory[i][4]
        error_cos_theta = math.cos(trajectory[i][2] - trajectory[i][5])  # trajectory[i][2]
        error_sin_theta = math.sin(trajectory[i][2] - trajectory[i][5])  # trajectory[i][2]

        error_x_rotated = np.cos(theta_transform) * error_x + np.sin(theta_transform) * error_y
        error_y_rotated = -np.sin(theta_transform) * error_x + np.cos(theta_transform) * error_y
        # error_x_rotated = error_x
        # error_y_rotated = error_y

        data = torch.FloatTensor([error_x_rotated, error_y_rotated, error_cos_theta, error_sin_theta])
        u = controller(data)
        vr = u[0].item()
        delta = u[1].item()

        init = trajectory[i]
        r = ode(vehicle_dynamics)
        r.set_initial_value(init)
        r.set_f_params([vr, delta, ref_input[0], ref_input[1]])
        val = r.integrate(r.t + time_step)

        t += time_step
        i += 1
        #  print(i,idx,u,res)
        trajectory.append(val)
        time.append(t)

        trace.append([t])
        trace[i].extend(val[3:])  # remove the reference trajectory from the trace
    return trace
'''
def TC_Simulate(waypoint: Waypoint, time_step: float,
                initial_point: np.array) -> np.array:
    mode: str = waypoint.mode
    mode_parameters: Optional[List[float]] = waypoint.mode_parameters
    time_bound = waypoint.time_bound
    if mode == 'follow_waypoint':
        ref_vx = (mode_parameters[3] - mode_parameters[0]) / time_bound
        ref_vy = (mode_parameters[4] - mode_parameters[1]) / time_bound
        # _, sym_rot_angle, sym_rot_angle_yz  = get_transform_information(waypoint)
        trace = runModel(mode_parameters[0:2] + list(initial_point), time_bound, time_step,
                         [ref_vx, ref_vy])
        return np.array(trace)
    else:
        raise ValueError("Mode: ", mode, "is not defined for the vehicle")
'''

# mode, initial_state, time_step, time_bound, x_d
def TC_Simulate(Mode,initialCondition,time_bound):
    # print("TC simulate")
    time_step = 0.01;
    time_bound = float(time_bound)
    Mode = Mode[1:-1]
    mode_parameters = Mode.split(";")
    mode_parameters = [float(x) for x in mode_parameters]
    number_points = int(np.ceil(time_bound / time_step))
    t = [i * time_step for i in range(0, number_points)]
    if t[-1] != time_step:
        t.append(time_bound)
    newt = []
    for step in t:
        newt.append(float(format(step, '.4f')))
    t = np.array(newt)

    ref_vx = (mode_parameters[2] - mode_parameters[0]) / time_bound
    ref_vy = (mode_parameters[3] - mode_parameters[1]) / time_bound
    ref_theta = np.arctan2(ref_vy, ref_vx)
    # _, sym_rot_angle = get_transform_information(waypoint)
    trace = runModel(mode_parameters[0:2] + [ref_theta] + list(initialCondition), time_bound, time_step,
                     [ref_vx, ref_vy, ref_theta])
    # print("Dryvr trace: ", trace)
    # trace = runModel(mode_parameters[0:2] + [0] + list(initialCondition), time_bound, time_step, [ref_vx, ref_vy])
    return np.array(trace)

# mode, initial_state, time_step, time_bound, x_d
def TC_Simulate_Batch(Mode, initialConditions,time_bound):
    # print('TC_Simulate_Batch',initialConditions)
    return np.stack(tuple(map(lambda x: TC_Simulate(Mode, x, time_bound), initialConditions)))


def get_transform_information(waypoint: Waypoint) -> Tuple[np.array, float]:
    mode: str = waypoint.mode
    mode_parameters: Optional[List[float]] = waypoint.mode_parameters
    time_bound = waypoint.time_bound
    if mode != "follow_waypoint":
        raise NotImplementedError("haven't implemented modes other than follow waypoint for these dynamics")
    # old_center = prev_mode_parameters
    dot = (mode_parameters[2] - mode_parameters[0])
    det = (mode_parameters[3] - mode_parameters[1])
    dir_angle = math.atan2(det, dot)
    # ref_vx = (mode_parameters[3] - mode_parameters[0]) / time_bound
    # ref_vy = (mode_parameters[4] - mode_parameters[1]) / time_bound
    translation_vector: np.array = np.zeros((3,))
    translation_vector[:2] = -1 * np.array(mode_parameters[2:])
    translation_vector[2] = -dir_angle
    return translation_vector, -dir_angle

# def get_transform_information(mode_parameters: List[float], initset: pc.Polytope) -> Tuple[np.array, float]:
#    pass


def transform_poly_to_virtual(poly, transform_information):
    translation_vector, new_system_angle = transform_information
    poly_out: pc.Polytope = poly.translation(translation_vector)
    # box = PolyUtils.get_bounding_box(poly_out)
    # poly_out = pc.box2poly(box.T)
    return  poly_out.rotation(i=0, j=1, theta=new_system_angle)


def transform_mode_to_virtual(waypoint: Waypoint, transform_information):
    point = waypoint.mode_parameters
    xs1 = point[0]  # x_i
    ys1 = point[1]  # y_i
    xd1 = point[2]
    yd1 = point[3]
    translation_vector, sc = transform_information
    x_n = translation_vector[0]
    y_n = translation_vector[1]
    xs2 = (xs1 + x_n) * math.cos(sc) - (ys1 + y_n) * math.sin(sc) # xs1 + x_n #
    ys2 = (xs1 + x_n) * math.sin(sc) + (ys1 + y_n) * math.cos(sc) #  ys1 + y_n #
    xd2 = (xd1 + x_n) * math.cos(sc) - (yd1 + y_n) * math.sin(sc) # xd1 + x_n #
    yd2 = (xd1 + x_n) * math.sin(sc) + (yd1 + y_n) * math.cos(sc) # yd1 + y_n #
    xs2 = round(xs2)
    ys2 = round(ys2)
    xd2 = round(xd2)
    yd2 = round(yd2)
    return Waypoint(waypoint.mode, [xs2, ys2,  xd2, yd2], waypoint.time_bound, waypoint.id)


def transform_poly_from_virtual(poly, transform_information):
    new_system_angle = -1 * transform_information[1]
    translation_vector = -1 * transform_information[0]
    out_poly = poly.rotation(i=0, j=1, theta=new_system_angle)
    return out_poly.translation(translation_vector)


def transform_mode_from_virtual(waypoint: Waypoint, transform_information):
    point = waypoint.mode_parameters
    sc = -1 * transform_information[1]
    translation_vector = -1 * transform_information[0]
    xs1 = point[0]  # x_i
    ys1 = point[1]  # y_i
    xd1 = point[3]
    yd1 = point[4]
    x_n = translation_vector[0]
    y_n = translation_vector[1]
    xs2 = (xs1) * math.cos(sc) - (ys1) * math.sin(sc) + x_n # xs1 + x_n #
    ys2 = (xs1) * math.sin(sc) + (ys1) * math.cos(sc) + y_n # ys1 + y_n #
    xd2 = (xd1) * math.cos(sc) - (yd1) * math.sin(sc) + x_n # xd1 + x_n #
    yd2 = (xd1) * math.sin(sc) + (yd1) * math.cos(sc) + y_n # yd1 + y_n #
    xs2 = round(xs2)
    ys2 = round(ys2)
    xd2 = round(xd2)
    yd2 = round(yd2)
    return Waypoint(waypoint.mode,[xs2, ys2, xd2, yd2], waypoint.time_bound, waypoint.id)


def transform_state_from_then_to_virtual_dryvr_string(point, transform_information_from, transform_information_to):
    pass


def get_virtual_mode_parameters():
    pass


def get_flowstar_parameters(mode_parameters: List[float], initial_set: np.array, time_step: float, time_bound: float,
                            mode: str):
    if mode != "follow_waypoint":
        raise NotImplementedError("These linear dynamics only support waypoint following mode")
    num_vars = 4
    order = 4
    hyper_params = [str(num_vars), str(time_step), str(time_bound), str(order)]
    cur_list: List[str] = hyper_params[:]
    var_names: List[str] = ["t", "x", "y", "z"]
    cur_list.extend(var_names[:num_vars])
    ode_rhs = ["1"]
    ode_rhs.extend([' + '.join([str(val) + " * (" + var_names[ind + 1] + ' - ' + "(" + str(mode_parameters[ind]) + "))"
                                for ind, val in enumerate(A_row)]) for A_row in A])
    cur_list.extend(ode_rhs)
    # time initset lowerbound
    cur_list.append('0')
    cur_list.extend([str(val) for val in initial_set[0, :]])
    # time initset upperbound
    cur_list.append('0')
    cur_list.extend([str(val) for val in initial_set[1, :]])
    return cur_list


def get_sherlock_parameters(mode_parameters: List[float], initial_set: np.array, time_step: float, time_bound: float,
                            mode: str):
    if mode != "follow_waypoint":
        raise NotImplementedError("These quadrotor dynamics only support waypoint following mode")
    num_nn_outputs = 1
    num_nn_inputs = initial_set.shape[1]
    num_vars = num_nn_inputs + num_nn_outputs
    order = 4
    nn_ctrl_file_path = "../systems_with_networks/Ex_Quadrotor/trial_controller_3"
    hyper_params = [str(num_vars), str(time_step), str(time_bound), str(order), nn_ctrl_file_path, num_nn_inputs,
                    num_nn_outputs]
    cur_list: List[str] = hyper_params[:]
    var_names: List[str] = ["t", "x", "y", "z"]
    cur_list.extend(var_names[:num_vars])
    ode_rhs = ["1"]
    ode_rhs.extend([' + '.join([str(val) + " * (" + var_names[ind + 1] + ' - ' + "(" + str(mode_parameters[ind]) + "))"
                                for ind, val in enumerate(A_row)]) for A_row in A])
    cur_list.extend(ode_rhs)
    # time initset lowerbound
    cur_list.append('0')
    cur_list.extend([str(val) for val in initial_set[0, :]])
    # time initset upperbound
    cur_list.append('0')
    cur_list.extend([str(val) for val in initial_set[1, :]])
    return cur_list

def get_flowstar_parameters(mode_parameters: List[float], initial_set: np.array, time_step: float, time_bound: float, mode: str):
    initial_condition = {'x1':[-0.2,-0.2],
                        'x2':[-0.2,-0.2],
                        'x3':[0,0],
                        'x4':[0,0],
                        'x5':[0,0],
                        'x6':[0,0],
                        'x7':[0.2,0.2],
                        'x8':[-0.2,-0.2],
                        'x9':[0,0]}

    ref_vx = (mode_parameters[3] - mode_parameters[0]) / time_bound
    ref_vy = (mode_parameters[4] - mode_parameters[1]) / time_bound
    ref_vz = (mode_parameters[5] - mode_parameters[2]) / time_bound

    # Initial condition for verisig is ex,ey,ez,vx,vy,vz,x,y,z
    initial_condition['x1'][0] = initial_set[0, 0] - mode_parameters[0]
    initial_condition['x1'][1] = initial_set[1, 0] - mode_parameters[0]
    initial_condition['x2'][0] = initial_set[0, 1] - mode_parameters[1]
    initial_condition['x2'][1] = initial_set[1, 1] - mode_parameters[1]
    initial_condition['x3'][0] = initial_set[0, 2] - mode_parameters[2]
    initial_condition['x3'][1] = initial_set[1, 2] - mode_parameters[2]
    initial_condition['x4'][0] = initial_set[0, 3]
    initial_condition['x4'][1] = initial_set[1, 3]
    initial_condition['x5'][0] = initial_set[0, 4]
    initial_condition['x5'][1] = initial_set[1, 4]
    initial_condition['x6'][0] = initial_set[0, 5]
    initial_condition['x6'][1] = initial_set[1, 5]
    initial_condition['x7'][0] = initial_set[0, 0]
    initial_condition['x7'][1] = initial_set[1, 0]
    initial_condition['x8'][0] = initial_set[0, 1]
    initial_condition['x8'][1] = initial_set[1, 1]
    initial_condition['x9'][0] = initial_set[0, 2]
    initial_condition['x9'][1] = initial_set[1, 2]

    ref_vx = (mode_parameters[3]-mode_parameters[0])/time_bound
    ref_vy = (mode_parameters[4]-mode_parameters[1])/time_bound
    ref_vz = (mode_parameters[5]-mode_parameters[2])/time_bound
    dynamics_string = f"x1' == x4 - ({ref_vx}+0) & \nx2' == x5 - ({ref_vy}+0) & \nx3' == x6 - ({ref_vz}+0) & \nx4' == 9.81 * sin(u1) / cos(u1) & \nx5' == - 9.81 * sin(u2) / cos(u2) & \nx6' == u3 - 9.81 & \nx7' = x4 & \nx8' = x5 & \nx9' = x6 & \nclock' == 1 & \nt' == 1"

    return [initial_condition, dynamics_string]

def transform_trace_from_virtual():
    pass

if __name__ == "__main__":
    wp = Waypoint('follow_waypoint',[20.4142135623735, 0.4142135623721741, 20.4142135623735, 2.4142135623721668],1.0,0)
    plt.figure(1)
    plt.figure(2)
    plt.figure(3)
    plt.figure(4)
    for i in range(10):
        x_init = np.random.uniform(18.16421,18.64342)
        y_init = np.random.uniform(2.16421,2.66421)
        theta_init = np.random.uniform(-2.61799,-0.5236)
        trace = TC_Simulate(wp, 0.01, [x_init,y_init,theta_init])
        trace = np.array(trace)
        plt.figure(1)
        plt.plot(trace[:,1],trace[:,2])
        # plt.plot([4.164213562373057,4.664213562373057,4.664213562373057,4.164213562373057,4.164213562373057],
        #         [22.16421356237288, 22.16421356237288, 22.66421356237288, 22.66421356237288, 22.16421356237288])
        plt.figure(2)
        plt.plot(trace[:,0],trace[:,1])
        plt.figure(3)
        plt.plot(trace[:,0],trace[:,2])
        plt.figure(4)
        plt.plot(trace[:,0],trace[:,3])
    plt.show()

    