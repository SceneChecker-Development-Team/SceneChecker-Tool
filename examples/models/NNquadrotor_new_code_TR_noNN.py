import torch
from scipy.integrate import ode
import numpy as np
import polytope as pc
from typing import Optional, List, Tuple
import math
from src.Waypoint import Waypoint

def dynamics(t, state, control):
    roll = control[0]
    pitch = control[1]
    thrust = control[2]

    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    xref = state[6]
    yref = state[7]
    zref = state[8]

    bx = control[3]
    by = control[4]
    bz = control[5]

    dx = vx
    dy = vy
    dz = vz

    dxi = xref - x
    dyi = yref - y
    dzi = zref - z

    dvx = 9.81 * np.tan(pitch)
    dvy = -9.81 * np.tan(roll)
    dvz = thrust - 9.81

    dxref = bx
    dyref = by
    dzref = bz

    return [dx, dy, dz, dvx, dvy, dvz, dxref, dyref, dzref, dxi, dyi, dzi]

def run_simulation(init, mode_parameters, time_bound, time_step):
    trajectory = [np.array(init)]
    t = 0
    sc = mode_parameters[3]
    trace = [[t]]
    trace[0].extend(init[0:6])
    i = 0
    while(t<time_bound):
        state = trajectory[-1]

        x = state[0]
        y = state[1]
        z = state[2]
        vx = state[3]
        vy = state[4]
        vz = state[5]
        xref = state[6]
        yref = state[7]
        zref = state[8]
        xi = state[9]
        yi = state[10]
        zi = state[11]

        vxref = mode_parameters[0]
        vyref = mode_parameters[1]
        vzref = mode_parameters[2]

        ex = xref - x
        ey = yref - y
        ez = zref - z
        evx = vxref - vx
        evy = vyref - vy
        evz = vzref - vz

        tmp1 = ex * np.cos(sc) - ey * np.sin(sc)
        tmp2 = ex * np.sin(sc) + ey * np.cos(sc)
        ex = tmp1
        ey = tmp2

        tmp1 = evx * np.cos(sc) - evy * np.sin(sc)
        tmp2 = evx * np.sin(sc) + evy * np.cos(sc)
        evx = tmp1
        evy = tmp2

        tmp1 = xi * np.cos(sc) - yi * np.sin(sc)
        tmp2 = xi * np.sin(sc) + yi * np.cos(sc)
        xi = tmp1
        yi = tmp2

        pitch = 1*ex + 1*evx + 1*xi
        roll = -(1*ey + 1*evy + 1*yi)
        thrust = 10*ez + 10*evz + 1*zi

        r = ode(dynamics)
        r.set_initial_value(state)

        r.set_f_params([roll, pitch, thrust, vxref, vyref ,vzref])

        val = r.integrate(r.t + time_step)
        trajectory.append(val)
        t += time_step
        i += 1
        trace.append([t])
        trace[i].extend(val[0:6])

    trace = np.array(trace)
    return trace

# mode, initial_state, time_step, time_bound, x_d
def TC_Simulate(Mode,initialCondition,time_bound):
    time_step = 0.05;
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

    ref_vx = (mode_parameters[3] - mode_parameters[0]) / time_bound
    ref_vy = (mode_parameters[4] - mode_parameters[1]) / time_bound
    ref_vz = (mode_parameters[5] - mode_parameters[2]) / time_bound
    sym_rot_angle = 0
    initialCondition = initialCondition.tolist()
    init = initialCondition + mode_parameters[0:3]+[0,0,0]
    trace = run_simulation(init, [ref_vx, ref_vy, ref_vz, sym_rot_angle], time_bound, time_step)
    return np.array(trace)

# mode, initial_state, time_step, time_bound, x_d
def TC_Simulate_Batch(Mode, initialConditions,time_bound):
    return np.stack(tuple(map(lambda x: TC_Simulate(Mode, x, time_bound), initialConditions)))

def get_transform_information(waypoint: Waypoint) -> Tuple[np.array, float]:
    mode: str = waypoint.mode
    mode_parameters: Optional[List[float]] = waypoint.mode_parameters
    time_bound = waypoint.time_bound
    if mode != "follow_waypoint":
        raise NotImplementedError("haven't implemented modes other than follow waypoint for these dynamics")
    # old_center = prev_mode_parameters
    dot = (mode_parameters[3] - mode_parameters[0])
    det = (mode_parameters[4] - mode_parameters[1])
    # detz = (mode_parameters[5] - mode_parameters[2])
    dir_angle = math.atan2(dot, det)
    dir_angle_yz = 0 # math.atan2(det, detz)
    ref_vx = (mode_parameters[3] - mode_parameters[0]) / time_bound
    ref_vy = (mode_parameters[4] - mode_parameters[1]) / time_bound
    ref_vz = (mode_parameters[5] - mode_parameters[2]) / time_bound
    translation_vector: np.array = np.zeros((6,))
    translation_vector[:3] = -1 * np.array(mode_parameters[3:])
    # translation_vector[3:] = -1 * np.array([ref_vx, ref_vy, ref_vz])
    translation_vector[3:] = -1 * np.array([0, 0, 0])
    return translation_vector, dir_angle, dir_angle_yz

# def get_transform_information(mode_parameters: List[float], initset: pc.Polytope) -> Tuple[np.array, float]:
#    pass


def transform_poly_to_virtual(poly, transform_information):
    translation_vector, new_system_angle, new_system_angle_yz = transform_information
    poly_out: pc.Polytope = poly.translation(translation_vector)
    poly_out = poly_out.rotation(i=0, j=1, theta=new_system_angle)
    # poly_out.rotation(i=1, j=2, theta=new_system_angle_yz)
    return poly_out.rotation(i=3, j=4, theta=new_system_angle)
    #return poly_out.rotation(i=4, j=5, theta=new_system_angle_yz)


def transform_mode_to_virtual(waypoint: Waypoint, transform_information):
    point = waypoint.mode_parameters
    xs1 = point[0]  # x_i
    ys1 = point[1]  # y_i
    zs1 = point[2]
    xd1 = point[3]
    yd1 = point[4]
    zd1 = point[5]
    translation_vector, sc, sc_yz = transform_information
    x_n = translation_vector[0]
    y_n = translation_vector[1]
    z_n = translation_vector[2]
    xs2 = (xs1 + x_n) * math.cos(sc) - (ys1 + y_n) * math.sin(sc)
    ys2 = (xs1 + x_n) * math.sin(sc) + (ys1 + y_n) * math.cos(sc)
    xd2 = (xd1 + x_n) * math.cos(sc) - (yd1 + y_n) * math.sin(sc)
    yd2 = (xd1 + x_n) * math.sin(sc) + (yd1 + y_n) * math.cos(sc)
    zs3 = zs1 + z_n
    zd3 = zd1 + z_n
    #ys3 = ys2 * math.cos(sc_yz) - (zs1 + z_n) * math.sin(sc_yz)
    #zs3 = ys2 * math.sin(sc_yz) + (zs1 + z_n) * math.sin(sc_yz)
    #yd3 = yd2 * math.cos(sc_yz) - (zd1 + z_n) * math.sin(sc_yz)
    #zd3 = yd2 * math.sin(sc_yz) + (zd1 + z_n) * math.sin(sc_yz)
    xs2 = round(xs2)
    ys3 = round(ys2) # round(ys3)
    zs3 = round(zs3) # round(zs3)
    xd2 = round(xd2)
    yd3 = round(yd2) # round(yd3)
    zd3 = round(zd3) # round(zd3)
    return Waypoint(waypoint.mode,[xs2, ys3, zs3, xd2, yd3, zd3], waypoint.time_bound, waypoint.id)


def transform_poly_from_virtual(poly, transform_information):
    new_system_angle = -1 * transform_information[1]
    translation_vector = -1 * transform_information[0]
    out_poly = poly.rotation(i=3, j=4, theta=new_system_angle)
    out_poly = out_poly.rotation(i=0, j=1, theta=new_system_angle)
    return out_poly.translation(translation_vector) # out_poly


def transform_mode_from_virtual(waypoint: Waypoint, transform_information):
    point = waypoint.mode_parameters
    sc = -1 * transform_information[1]
    translation_vector = -1 * transform_information[0]
    xs1 = point[0]  # x_i
    ys1 = point[1]  # y_i
    zs1 = point[2]
    xd1 = point[3]
    yd1 = point[4]
    zd1 = point[5]
    x_n = translation_vector[0]
    y_n = translation_vector[1]
    z_n = translation_vector[2]
    xs2 = (xs1) * math.cos(sc) - (ys1) * math.sin(sc) + x_n
    ys2 = (xs1) * math.sin(sc) + (ys1) * math.cos(sc) + y_n
    xd2 = (xd1) * math.cos(sc) - (yd1) * math.sin(sc) + x_n
    yd2 = (xd1) * math.sin(sc) + (yd1) * math.cos(sc) + y_n
    zs3 = zs1 + z_n
    zd3 = zd1 + z_n
    #ys3 = (ys2) * math.cos(sc_yz) - (zs1) * math.sin(sc_yz)
    #zs3 = (ys2) * math.sin(sc_yz) + (zs1) * math.cos(sc_yz) + z_n
    #yd3 = (yd2) * math.cos(sc_yz) - (zd1) * math.sin(sc_yz)
    #zd3 = (yd2) * math.sin(sc_yz) + (zd1) * math.cos(sc_yz) + z_n
    xs2 = round(xs2)
    ys3 = round(ys2)  # round(ys3)
    zs3 = round(zs3)  # round(zs3)
    xd2 = round(xd2)
    yd3 = round(yd2)  # round(yd3)
    zd3 = round(zd3)  # round(zd3)
    return Waypoint(waypoint.mode,[xs2, ys3, zs3, xd2, yd3, zd3], waypoint.time_bound, waypoint.id)
    # x2 = x1 * math.cos(sc) + y1 * math.sin(sc) - x_n
    # y2 = -1 * x1 * math.sin(sc) + y1 * math.cos(sc) - y_n
    # x2 = x1 - x_n
    # y2 = y1 - y_n
    # return [x2, y2]


def transform_state_from_then_to_virtual_dryvr_string(point, transform_information_from, transform_information_to):
    pass


def get_virtual_mode_parameters():
    pass


def get_flowstar_parameters(mode_parameters: List[float], initial_set: np.array, time_step: float, time_bound: float, mode: str):

    num_variables = initial_set.shape[1]
    initial_condition = {'x':[0,0],
                        'y':[0,0],
                        'z':[0,0],
                        'vx':[0,0],
                        'vy':[0,0],
                        'vz':[0,0],
                        'xref':[0,0],
                        'yref':[0,0],
                        'zref':[0,0],
                        'xi':[0,0],
                        'yi':[0,0],
                        'zi':[0,0],
                        't':[0,0]}

    initial_condition['x'][0] = initial_set[0,0]
    initial_condition['y'][0] = initial_set[0,1]
    initial_condition['z'][0] = initial_set[0,2]
    initial_condition['vx'][0] = initial_set[0,3]
    initial_condition['vy'][0] = initial_set[0,4]
    initial_condition['vz'][0] = initial_set[0,5]
    initial_condition['xref'][0] = mode_parameters[0] 
    initial_condition['yref'][0] = mode_parameters[1] 
    initial_condition['zref'][0] = mode_parameters[2] 
    initial_condition['xi'][0] = 0
    initial_condition['yi'][0] = 0 
    initial_condition['zi'][0] = 0
    initial_condition['t'][0] = 0

    initial_condition['x'][1] = initial_set[1,0]
    initial_condition['y'][1] = initial_set[1,1]
    initial_condition['z'][1] = initial_set[1,2]
    initial_condition['vx'][1] = initial_set[1,3]
    initial_condition['vy'][1] = initial_set[1,4]
    initial_condition['vz'][1] = initial_set[1,5]
    initial_condition['xref'][1] = mode_parameters[0] 
    initial_condition['yref'][1] = mode_parameters[1] 
    initial_condition['zref'][1] = mode_parameters[2] 
    initial_condition['xi'][1] = 0
    initial_condition['yi'][1] = 0 
    initial_condition['zi'][1] = 0
    initial_condition['t'][1] = 0

    segvx = (mode_parameters[3]-mode_parameters[0])/time_bound
    segvy = (mode_parameters[4]-mode_parameters[1])/time_bound
    segvz = (mode_parameters[5]-mode_parameters[2])/time_bound

    time_step = 0.001
    dynamics_string = f"x' = vx\n\
                y' = vy\n\
                z' = vz\n\
                vx' = 9.81 * sin(1*(xref - x) + 1*({segvx}-vx) + 1*xi)/cos(1*(xref - x) + 1*({segvx}-vx) + 1*xi)\n\
                vy' = -9.81 * sin(-(1*(yref - y) + 1*({segvy}-vy) + 1*yi))/cos(-(1*(yref - y) + 1*({segvy}-vy) + 1*yi))\n\
                vz' = 10*(zref - z) + 10*({segvz}-vz) + 1*zi - 9.81\n\
                xref' = {segvx}\n\
                yref' = {segvy}\n\
                zref' = {segvz}\n\
                xi' = xref - x\n\
                yi' = yref - y\n\
                zi' = zref - z\n\
                t' = 1\n"

    model_string = f"continuous reachability\n\
    {{\n\
        state var x, y, z, vx, vy, vz, xref, yref, zref, xi, yi, zi, t\n\n\
        setting\n\
        {{\n\
            fixed steps {time_step}\n\
            time {time_bound}\n\
            remainder estimation 1e-2\n\
            identity precondition\n\
            gnuplot interval x,y\n\
            adaptive orders {{min 4,max 8}}\n\
            cutoff 1e-12\n\
            precision 53\n\
            output flowstar_tube\n\
            print off\n\
        }}\n\n\
        nonpoly ode {{20}}\n\
        {{\n\
            {dynamics_string}\
        }}\n\
        init\n\
        {{\n\
            x in [{initial_condition['x'][0]},{initial_condition['x'][1]}]\n\
            y in [{initial_condition['y'][0]},{initial_condition['y'][1]}]\n\
            z in [{initial_condition['z'][0]},{initial_condition['z'][1]}]\n\
            vx in [{initial_condition['vx'][0]},{initial_condition['vx'][1]}]\n\
            vy in [{initial_condition['vy'][0]},{initial_condition['vy'][1]}]\n\
            vz in [{initial_condition['vz'][0]},{initial_condition['vz'][1]}]\n\
            xref in [{initial_condition['xref'][0]},{initial_condition['xref'][1]}]\n\
            yref in [{initial_condition['yref'][0]},{initial_condition['yref'][1]}]\n\
            zref in [{initial_condition['zref'][0]},{initial_condition['zref'][1]}]\n\
            xi in [{initial_condition['xi'][0]},{initial_condition['xi'][1]}]\n\
            yi in [{initial_condition['yi'][0]},{initial_condition['yi'][1]}]\n\
            zi in [{initial_condition['zi'][0]},{initial_condition['zi'][1]}]\n\
            t in [{initial_condition['t'][0]},{initial_condition['t'][1]}]\n\
        }}\n\
    }}"

    return [model_string,num_variables]


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

def transform_trace_from_virtual(trace: np.array, transform_information):
    sc = -1*transform_information[1]
    translation_vector = -1*transform_information[0]
    x1 = trace[:,0:1]
    y1 = trace[:,1:2]
    z1 = trace[:,2:3]
    x_n = translation_vector[0]
    y_n = translation_vector[1]
    z_n = translation_vector[2]
    x2 = x1*math.cos(sc) - y1*math.sin(sc) + x_n
    y2 = x1*math.sin(sc) + y1*math.cos(sc) + y_n
    z3 = z1 + z_n
    res = np.concatenate((x2,y2,z3), axis=1)
    return res