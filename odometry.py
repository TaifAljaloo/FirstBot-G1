from motors import motor
from math import pi
import numpy as np
import matplotlib.pyplot as plt

L = 0.191
R = 0.026

def direct_kinematics(vg, vd):
    return R/2 * (vd + vg), R/L * (vd - vg)

def odom(vl, va, dt):
    eps = 0.001

    dtheta = va * dt

    if abs(va) > eps:
        rc = vl/va
        dx = rc * (1 - np.cos(dtheta))
        dy = rc * np.sin(dtheta)
    else:
        dx = vl * dt
        dy = 0
    
    return dx, dy, dtheta


def tick_odom(xn, yn, thetan, vl, va, dt):
    dx, dy, dtheta = odom(vl, va, dt)

    dxn = dx * np.cos(thetan) - dy*np.sin(thetan)
    dyn = dx * np.sin(thetan) + dy*np.cos(thetan)

    return xn + dxn, yn + dyn, thetan - dtheta

def inverse_kinematics(vl, va):
    return vl/R - (va * L)/(2 * R), vl/R + (va * L)/(2 * R)

def go_to_xya(x_target, y_target, theta_target, dist_tolerance=0.01, theta_tolerance=0.01):

    motor_control = motor()
    x, y, theta = 0, 0, 0
    motor_control.lock()
    dir_angle = np.arctan2(y_target - y, x_target - x) 
    rotate_robot(motor_control, dir_angle)
    motor_control.move(0.1, 0.1)   
    t0 = time.time()

    dist = np.sqrt((x_target - x)**2 + (y_target - y)**2)
    theta_error = theta_target - theta
    while dist > dist_tolerance or abs(theta_error) > theta_tolerance:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        x_error = x_target - x
        y_error = y_target - y
        theta_error = theta_target - theta

        dist = np.sqrt(x_error**2 + y_error**2)

        right_angular_speed, left_angular_speed = motor_control.get_speed()
        right_angular_speed *= -1

        linear_speed, angular_speed = direct_kinematics(left_speed, right_speed)

        x, y, theta = tick_odom(x, y, theta, linear_speed, angular_speed, dt)

        dir_angle = np.arctan2(y_target - y, x_target - x)

        motor_control.move(0.2 * dist + 0.1, 1.2 * (dir_angle - theta))

    motor_control.stop()
    motor_control.lock()

    rotate_robot(motor_control, theta_target)

def print_map():

    with open("logs_1.txt", "r") as f: data = [i.replace("\n", "").split(" ") for i in f.readlines()]

    l_data, l_pos = data[1], [0,0,0]
    c_color = data[0][1]
    col = {c_color: []}

    for i in range(2, len(data)):
        if data[i][0] == "#":
            c_color = data[i][1]
            col[c_color] = []
        else:
            vl, va = direct_kinematics(float(data[i][0]), float(data[i][1]))
            dx, dy, dtheta = tick_odom(l_pos[0], l_pos[1], l_pos[2], vl, va, float(data[i][2]) - float(l_data[2]))
            col[c_color].append([dx, dy, dtheta])
            l_data, l_pos = data[i], [dx, dy, dtheta]

    for c in col.keys():
        plt.scatter([p[0] for p in col[c]], [p[1] for p in col[c]], color=c)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajet du robot')
    plt.axis('equal')
    plt.show()

print_map()