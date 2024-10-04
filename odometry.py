from motors import motor
from math import pi
import numpy as np
import time
import matplotlib.pyplot as plt

L = 0.191
R = 0.026

GOTO_SPEED = 17
REFRESH_RATE = 0.01

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

def rotate_robot(motor_control, target_angle):
    speed = GOTO_SPEED
    distance_to_rotate = target_angle
    motor_control.move(speed * np.sign(target_angle), -speed * np.sign(target_angle))
    current_rotate = 0
    print(target_angle)
    while abs(current_rotate) < abs(distance_to_rotate):
        left_angular_speed, right_angular_speed = motor_control.get_speed()
        #convert to rad
        vl,va = direct_kinematics(left_angular_speed * pi / 180, -right_angular_speed * pi / 180)
        current_rotate += va * REFRESH_RATE
        time.sleep(REFRESH_RATE)
    motor_control.stop()



def go_to_xya(x_target, y_target, theta_target, dist_tolerance=0.01, theta_tolerance=0.01):
    speed=GOTO_SPEED
    motor_control = motor()
    x, y, theta = 0, 0, 0

    motor_control.lock()
    dir_angle = np.arctan2(y_target - y, x_target - x) 
    rotate_robot(motor_control, dir_angle)
    
    distance_to_move = np.sqrt((x_target - x)**2 + (y_target - y)**2)
    distance_moved = 0
    motor_control.move(speed, speed)   

    while distance_moved < distance_to_move:
    
        left_angular_speed, right_angular_speed = motor_control.get_speed()
        vl,va = direct_kinematics(abs(left_angular_speed * pi / 180), abs(right_angular_speed * pi / 180))
        distance_moved += vl * REFRESH_RATE
        time.sleep(REFRESH_RATE)

    
    motor_control.stop()
    

    rotate_robot(motor_control, theta_target-dir_angle)

