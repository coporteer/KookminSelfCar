#!/usr/bin/env python3
# -*- coding: utf-8 -*- 16
import numpy as np
from numpy.linalg import inv

class KalmanPos2Vel():
    def __init__(self, P0, x0, q1=5, q2=1, r=10, dt=0.01):
        """칼만 필터 클래스 초기화 함수"""
        self.A = np.array([[1, dt],
                           [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = np.array([[q1, 0],
                           [0, q2]])
        self.R = np.array([[r]])
        self.P = P0
        self.dt = dt
        self.x_esti = x0

    def update(self, z_meas):
        x_pred = np.matmul(self.A, self.x_esti)
        P_pred = np.matmul(np.matmul(self.A, self.P), self.A.T) + self.Q
        K = np.matmul(P_pred,np.matmul(self.H.T, inv(np.matmul(np.matmul(self.H,P_pred), self.H.T) + self.R)))
        self.x_esti = x_pred + np.matmul(K, (z_meas - np.matmul(self.H, x_pred)))
        self.P = P_pred - np.matmul(K, np.matmul(self.H, P_pred))

        return self.x_esti, self.P