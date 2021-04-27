# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

import math
from numpy.linalg import inv

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 3.32
        self.lf = 1.01
        self.Ca = 20000
        self.Iz = 29526.2
        self.m = 4500
        self.g = 9.81

        # Add additional member variables according to your need here.
        # constraints
        self.F_max = 16000.0
        self.F_min = 0.0
        self.delta_min = -math.pi / 6
        self.delta_max = math.pi / 6

        # pid params
        self.kp_x = 50000.0
        self.ki_x = 150.0
        self.kd_x = 100.0
        self.kp_psi = 5.0
        self.ki_psi = 0.1
        self.kd_psi = 0.3

        #
        self.sum_error_x = 0.0
        self.error_x_old = 0.0
        self.sum_error_psi = 0.0
        self.error_psi_old = 0.0

        self.long_look_ahead = 650
        self.lat_look_ahead = 80

        # lateral
        self.track_center = np.average(trajectory, axis=0)

        self.error_state = np.array([[0.0], [0.0], [0.0], [0.0]])
        self.delta = 0.0
        self.F = 0.0

        # LQR gains (computed in MATLAB)
        self.K = np.array([[0.0057, 0.0234, 0.1303, 0.1265],
                           [0, 0, 0, 0]])
        # self.K = np.array([[0.001, 0.0001, 1.0, 0.1],
        #                    [0.0, 0.0, 0.0, 0.0]])
        self.delta_last = -100.0

        self.XTE_straight = 0.0
        self.XTE_small_angle = 0.0
        self.XTE_medium_angle = 0.0
        self.XTE_large_angle = 0.0
        self.XTE_super_large_angle = 0.0
        self.cnt_straight = 0.001
        self.cnt_small_angle = 0.001
        self.cnt_medium_angle = 0.001
        self.cnt_large_angle = 0.001
        self.cnt_super_large_angle= 0.001

    def inertial2global(self, x, y, psi_):
        # convert (x, y) from inertial frame to global frame
        # psi_ = wrapToPi(psi)
        xy_inertial = np.array([[x],
                                [y]])
        convert_mat = np.array([[math.cos(psi_), -math.sin(psi_)],
                                [math.sin(psi_), math.cos(psi_)]])
        XY_global = np.matmul(convert_mat, xy_inertial)
        return XY_global[0][0], XY_global[1][0]

    def global2inertial(self, X, Y, psi):
        # convert (X, Y) from global frame to inertial frame
        # psi_ = wrapToPi(psi)
        XY_global = np.array([[X],
                              [Y]])
        convert_mat = np.array([[math.cos(psi), -math.sin(psi)],
                                [math.sin(psi), math.cos(psi)]])
        convert_mat = np.linalg.inv(convert_mat)

        xy_inertial = np.matmul(convert_mat, XY_global)
        return xy_inertial[0][0], xy_inertial[1][0]

    def wrapAngle(self, theta):
        return (theta + 2 * math.pi) % (2 * math.pi)

    def performance_printout(self):
        print("[straight, small, med, large, superlarge] = [",
              self.XTE_straight/self.cnt_straight, ", ",
              self.XTE_small_angle/self.cnt_small_angle, ", ",
              self.XTE_medium_angle/self.cnt_medium_angle, ", ",
              self.XTE_large_angle/self.cnt_large_angle, ", ",
              self.XTE_super_large_angle/self.cnt_super_large_angle, "]")
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below.
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta).

        # preprocessing the reference trajectory
        # lateral preprocessing
        long_look_ahead = self.long_look_ahead
        ### !!! change this
        lat_look_ahead = self.lat_look_ahead
        XTE, nn_idx = closestNode(X, Y, trajectory)
        nn_lat_next_idx = nn_idx + lat_look_ahead
        if nn_lat_next_idx >= len(trajectory) - 1:
            # print("lat near end")
            nn_lat_next_idx = len(trajectory) - 1

        X_next_ref = 0.0
        Y_next_ref = 0.0
        cnt = 0
        for i in range(nn_idx, nn_lat_next_idx):
            cnt += 1
            X_next_ref += trajectory[nn_lat_next_idx][0]
            Y_next_ref += trajectory[nn_lat_next_idx][1]
        X_next_ref /= cnt
        Y_next_ref /= cnt
        # X_next_ref = trajectory[nn_lat_next_idx][0]
        # Y_next_ref = trajectory[nn_lat_next_idx][1]
        psi_ref = math.atan2(Y_next_ref - Y, X_next_ref - X)
        speed_scale = 1.0
        longi_scale = 1.0

        # longitude lookahead
        #   1. comparing with current psi, to determine if there is a curb ahead
        #   2. generate reference xdot, for longitudinal controller
        nn_long_next_idx = nn_idx + long_look_ahead
        if nn_long_next_idx >= len(trajectory) - 1:
            long_look_ahead = len(trajectory) - 1 - nn_idx
            nn_long_next_idx = len(trajectory) - 1
        X_long_next_ref = trajectory[nn_long_next_idx][0]
        Y_long_next_ref = trajectory[nn_long_next_idx][1]
        Xdot_ref = (X_long_next_ref - X) / (delT * long_look_ahead)
        Ydot_ref = (Y_long_next_ref - Y) / (delT * long_look_ahead)
        xdot_ref, ydot_ref = self.global2inertial(Xdot_ref, Ydot_ref, psi)
        # state machine
        # straight line boost
        psi_long_ref = math.atan2(Y_long_next_ref - Y, X_long_next_ref - X)
        error_psi_long = self.wrapAngle(psi_long_ref) - self.wrapAngle(psi)

        if np.abs(error_psi_long) < 20 * math.pi / 180:  # straight
            # print("straight!")
            self.cnt_straight += 1
            self.XTE_straight += XTE

            longi_scale = 4.0
            self.kd_x = 5.0
            self.lat_look_ahead = 60
        elif np.abs(error_psi_long) < 30 * math.pi / 180:  # curb
            # print("small angle is", np.abs(error_psi_long)*180/math.pi)
            self.cnt_small_angle += 1
            self.XTE_small_angle += XTE

            longi_scale = 0.8
            self.kd_x = 2.0
            self.lat_look_ahead = 100
        elif np.abs(error_psi_long) < 45 * math.pi / 180:  # medium
            # print("median angle is", np.abs(error_psi_long)*180/math.pi)
            self.cnt_medium_angle += 1
            self.XTE_medium_angle += XTE

            longi_scale = 0.5
            self.kd_x = 0.0
            self.long_look_ahead = 650
            self.lat_look_ahead = 125
        elif np.abs(error_psi_long) < 55 * math.pi / 180:  # medium
            # print("median-large angle is", np.abs(error_psi_long)*180/math.pi)
            self.cnt_medium_angle += 1
            self.XTE_medium_angle += XTE

            longi_scale = 0.5
            self.kd_x = 0.0
            self.long_look_ahead = 650
            self.lat_look_ahead = 150
        elif np.abs(error_psi_long) < 85 * math.pi / 180:  # curb
            # print("large angle is", np.abs(error_psi_long)*180/math.pi)
            self.cnt_large_angle += 1
            self.XTE_large_angle += XTE

            longi_scale = 0.5
            self.kd_x = 0.0
            self.long_look_ahead = 650
            self.lat_look_ahead = 200
        else:
            # print("super large angle is", np.abs(error_psi_long)*180/math.pi)
            self.cnt_super_large_angle += 1
            self.XTE_super_large_angle += XTE

            longi_scale = 0.5
            self.kd_x = 0.0
            self.long_look_ahead = 650
            self.lat_look_ahead = 200
        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        # generate error state for lateral controller
        # compute e1, e2 e1dot e2dot
        # compute e1
        e1 = np.sqrt((X - X_next_ref)**2 + (Y - Y_next_ref)**2)
        # e1 = XTE
        XY_center = np.sqrt((X - self.track_center[0])**2 + (Y - self.track_center[1])**2)
        XY_ref_center = np.sqrt((trajectory[nn_idx][0] - self.track_center[0])**2 + (trajectory[nn_idx][1] - self.track_center[1])**2)
        # XY_ref_center = np.sqrt((X_next_ref - self.track_center[0])**2 + (Y_next_ref - self.track_center[1])**2)

        if XY_ref_center > XY_center:
            # print("vehicle inside[car, ref]", XY_center, " ",  XY_ref_center, "; ", self.lat_look_ahead, "; speed ", xdot)
            e1 *= -1
        # else:
        #     print("vehicle outside[car, ref]", XY_center, " ", XY_ref_center, "; ", self.lat_look_ahead, "; speed ", xdot)
        # compute e2
        # print("XTE=", XTE, "; lookahead=", self.lat_look_ahead)
        e2 = - self.wrapAngle(psi) + self.wrapAngle(psi_ref)
        e2 = wrapToPi(e2)
        # compute e1dot
        e1dot = (e1 - self.error_state[0][0])/0.2
        # compute e2dot
        e2dot = (e2 - self.error_state[2][0])/0.2
        # print("[psi, psi ref] = ", psi, " ", psi_ref)
        # print("states: ", e1, " ", e2, " ", e1dot, " ", e2dot)
        # form the error state
        self.error_state[0][0] = e1
        self.error_state[1][0] = e1dot
        self.error_state[2][0] = -e2
        self.error_state[3][0] = -e2dot

        # solve DT LQR tracking problem
        # K are computed in MATLAB

        # get controls
        sys_control_next = np.matmul(self.K, self.error_state)
        delta = - sys_control_next[0][0]
        delta = clamp(delta, self.delta_min, self.delta_max)
        # print(delta)
        if self.delta_last == -100.0:
            print("first timestep")
            self.delta_last = delta
        else:
            if abs(e2)<0.03:
                # print("clamp 0.03!", abs(e2))
                delta = clamp(delta, self.delta_last-0.025, self.delta_last+0.025)
            elif abs(e2)<0.05:
                # print("clamp! 0.05", abs(e2))
                delta = clamp(delta, self.delta_last-0.04, self.delta_last+0.04)
            elif abs(e2)<0.08:
                # print("clamp! 0.8", abs(e2))
                delta = clamp(delta, self.delta_last-0.07, self.delta_last+0.07)
            else:
                # print("clamp! over", abs(e2))
                delta = clamp(delta, self.delta_last-0.1, self.delta_last+0.1)
            self.delta_last = delta

        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        """
        error_x = xdot_ref * speed_scale * longi_scale - xdot
        self.sum_error_x += error_x * delT
        F = self.kp_x * error_x + \
            self.ki_x * self.sum_error_x + \
            self.kd_x * (error_x - self.error_x_old) / delT
        F = clamp(F, self.F_min, self.F_max)
        # # print("longi force: ", F, "XTE= ", XTE)
        # self.error_x_old = error_x
        # print("ref v= ", xdot_ref, "; real v= ", xdot)
        # print("ref angle= ", psi_ref, "; real angle= ", psi)
        # print("lateral angle= ", delta, "; longi force= ", F, "; XTE= ", XTE)

        # performance printout
        # self.performance_printout()

        # print(e1, " ", e1dot, " ", e2, " ", e2dot, " ", F, " ", delta)
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta

