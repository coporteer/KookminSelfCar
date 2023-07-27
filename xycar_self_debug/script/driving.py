#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이 파일은 과제 2의 자율 주행을 위한 코드
차선 인식 클래스 LaneDetector를 이용하여 차선을 인식하고
PID 제어 및 Kalman 필터를 이용하여 차량을 주행함
"""
#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2
import rospy, time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from math import *
import signal
import sys
import os
import time
from numpy.linalg import inv

# LaneDetector 파일에서 LaneDetector 클래스 불러오기
from LaneDetector import LaneDetector

class PID():
    """ PID 제어 동작을 수행
        PID의 정의에 따라 비례동작(P)값은 상수(Kp)를 곱해주고 미분동작(D)값은 OverShoot현상에 대응하기위해 상수(Kd)를 
        직전 error 값을 비교하여 곱해 값을 출력한다. 후에 이를 모두 더해 비례미분동작(PD) 수행 
    """
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, dt=0.01, max_u = 50):
        """ PID 컨트롤러에 필요한 파라미터 값 초기화
        """
        self.Kp = Kp # 비례 gain
        self.Ki = Ki # 적분 gain
        self.Kd = Kd # 미분 gain
        self.dt = dt # 시간 간격
        self.max_u = max_u # 최대 제어 신호 크기. 시뮬레이터에서 최대 조향각

        self.i_err = 0 # error의 적분값
        self.prev_err = 0 # 이전의 error 값

    def do(self, r, y, dt): # tracking에서 angle값을 구하기 위해 호출됩
        """ PID 제어의 본 기능 구현 코드입니다.
            오차는 가장 가까운 경로의 각도와 현재 각도의 차이
            pid 제어 연산을 통해 u 를 구한 뒤, 최대최소 조향각 보정 후 반환

        Args :
            r (int): 차량의 현재 위치와 가장 가까운 경로로의 목표 각도
            y (numpy.float64): 차량의 주행 중 현재 각도
            dt (numpy.float64): 단위 시간

        Returns :
            u (numpy.float64) : pid 제어를 통해 최종적으로 연산된 조향각

        """
        err = r - y # 오차값 = r(target_yaw) - y(yaw) 
        
        # PID 제어의 각 항들을 계산
        up = self.Kp * err # 비례항. 현재 에러값을 곱함
        ui = self.Ki * self.i_err # 적분항. 에러의 적분값을 곱함
        ud = self.Kd * (err - self.prev_err) / self.dt # 미분항. 이전 에러값을 곱한 뒤 dt로 나눠 미분을 표현함

        u = up + ui + ud # pid 제어 값들을 더해 전체 제어 값을 구함

        # 시뮬레이션의 최대, 최소 조향각 사이로 구현하기 위한 코드
        u = max(min(u, self.max_u), -self.max_u) # u가 50보다 큰 경우, max_u(50)이 u에 저장되고, -50보다 작은 경우, -max_u가 저장됨
        self.prev_err = err # 다음 angle값을 구하기 위해 현재의 오차값을 prev_err로 저장
        self.i_err += err * self.dt # 적분항에 사용하기 위해 현재 오차값에 dt를 곱해 더함
        
        return u 

class KalmanPos2Vel():
    """칼만 필터 클래스
    
    """
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
        """ 상태 변수 추정을 위한 칼만필터 업데이트 함수

        Args:
            z_meas (int): 필터링 대상

        Return:
            self.x_esti (numpy.array): 추정된 상태 변수 x
            self.P (numpy.array): 

        """
        x_pred = np.matmul(self.A, self.x_esti)
        P_pred = np.matmul(np.matmul(self.A, self.P), self.A.T) + self.Q
        K = np.matmul(P_pred,np.matmul(self.H.T, inv(np.matmul(np.matmul(self.H,P_pred), self.H.T) + self.R)))
        self.x_esti = x_pred + np.matmul(K, (z_meas - np.matmul(self.H, x_pred)))
        self.P = P_pred - np.matmul(K, np.matmul(self.H, P_pred))

        return self.x_esti, self.P


def signal_handler(sig, frame):
    """ 터미널에서 Ctrl-c 키입력으로 프로그램 실행을 끝낼 때 그 처리시간을 줄이기 위한 함수
    """
    time.sleep(3)
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0]) # 카메라 이미지를 담을 변수
bridge = CvBridge()
motor = None # 모터 토픽을 담을 변수

#=============================================
# 프로그램에서 사용할 상수 선언부
#=============================================
CAM_FPS = 30    # 카메라 FPS - 초당 30장의 사진을 보냄
WIDTH, HEIGHT = 640, 480    # 카메라 이미지 가로x세로 크기


def img_callback(data):
    """ 콜백함수 - 카메라 토픽을 처리하는 콜백함수
    카메라 이미지 토픽이 도착하면 자동으로 호출되는 함수
    토픽에서 이미지 정보를 꺼내 image 변수에 옮겨 담음.
    """
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

def drive(angle, speed):
    """ 모터 토픽을 발행하는 함수
    입력으로 받은 angle과 speed 값을 모터 토픽에 옮겨 담은 후에 토픽을 발행함
    """

    global motor

    motor_msg = xycar_motor()
    motor_msg.angle = angle
    motor_msg.speed = speed

    motor.publish(motor_msg)


#=============================================
# 실질적인 메인 함수
# 카메라 토픽을 받아 각종 영상처리와 알고리즘을 통해
# 차선의 위치를 파악한 후에 조향각을 결정하고,
# 최종적으로 모터 토픽을 발행하는 일을 수행함.
#=============================================
def start():
    """ 실질적인 메인 함수
    카메라 토픽을 받아 각종 영상처리와 알고리즘을 통해 차선의 위치를 파악한 후에 조향각을 결정하고,
    최종적으로 모터 토픽을 발행하는 일을 수행함.
    """

    # 위에서 선언한 변수를 start() 안에서 사용하고자 함
    global motor, image


    rospy.init_node('driving') # ROS 노드를 생성하고 초기화 함.
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1) # 모터 토픽을 발행할 것임을 선언
    rospy.Subscriber("/usb_cam/image_raw/",Image, img_callback) # 카메라 토픽을 구독

    print ("----- Xycar self driving -----")

    # 첫번째 카메라 토픽이 도착할 때까지 기다림.
    rospy.wait_for_message("/usb_cam/image_raw/", Image)

    # 변수 선언
    xsize, ysize = WIDTH, HEIGHT
    pt_src = np.float32([(500, 300),
                        (160, 300), 
                        (32, 410), 
                        (630, 410)]) # perspective transformation의 원본 좌표
    pt_dst = np.float32([
        (xsize - 120, 0),
        (120, 0),
        (120, ysize),
        (xsize - 120, ysize)
    ]) # perspective transformation 후 변환된 좌표
    roi_vertices = np.array([[[50, ysize],
                              [50, 0],
                              [xsize-50, 0],
                              [xsize-50, ysize]]], dtype=np.int32)  # roi 좌표
    dt = 1 / 30.0 # 30 FPS
    angle = 0
    previous_lane = None
    CENTER_TO_LANE_PIXELS = 161         # 차가 차선 중앙에 위치할 때 두 차선의 너비의 반 (pixel 단위))
    x_center, x_yaw = 320, -90          # 차량의 현재 횡방향 위치와 방향각
    ref_center, ref_yaw = 320, -90      # 제어 목표 값

    # 차선 인식 알고리즘 수행
    ld = LaneDetector((xsize, ysize), pt_src, pt_dst, roi_vertices = roi_vertices)

    # PID 객체 생성 
    pid_center = PID(Kp=0.15, Ki=0.00, Kd=0.05, dt=dt, max_u =10)
    pid_yaw = PID(Kp=0.65, Ki=0.0, Kd=0.0, dt=dt, max_u = 10)
    kf_center = KalmanPos2Vel(x0=np.array([WIDTH//2,0]), P0=5*np.eye(2), q1=2, q2=1, r=10, dt=dt)
    kf_yaw = KalmanPos2Vel(x0=np.array([-90,0]), P0=5*np.eye(2), q1=2, q2=1, r=10, dt=dt)
    rate = rospy.Rate(int(1/dt))

    #=========================================
    # 메인 루프
    # 카메라 토픽이 도착하는 주기에 맞춰 한번씩 루프를 돌면서
    # "이미지처리 + 차선위치찾기 + 조향각 결정 + 모터토픽 발행"
    # 작업을 반복적으로 수행함.
    #=========================================
    while not rospy.is_shutdown():
        # 이미지 처리를 위해 카메라 원본 이미지를 img에 복사 저장한다.
        img = image.copy()
        ret, img, output = ld.do(image)

        if ret <= LaneDetector.SUCCESS_POLY_FIT_ONE:
            # 최소한 하나 이상의 차선을 찾은 경우

            # 두 개의 차선을 찾았을 때 기존 프레임에서 찾았던 차선을 가능하면 선택하기 위한 구문
            if previous_lane == 'left':
                # 지난 프레임에서 왼쪽 차선을 찾았으면 현재 프레임에서 왼쪽 차선이 보이는지 먼저 확인
                selected_lane = 'left' if output[0] != -1 else 'right'
            elif previous_lane == 'right':
                # 지난 프레임에서 오른쪽 차선을 찾았으면 현재 프레임에서 오른쪽 차선이 보이는지 먼저 확인
                selected_lane = 'right' if output[1] != -1 else 'left'
            else:
                # 지난 프레임에서 찾은 차선이 없다면 가능하면 왼쪽을 먼저 선택
                selected_lane = 'left' if output[0] != -1 else 'right'
            previous_lane = selected_lane

            # 하나의 선택된 차선을 기준으로 차량의 횡방향 위치와 방향각을 결정
            if selected_lane == 'left':
                x_center = output[0] + CENTER_TO_LANE_PIXELS
                x_yaw = output[2]
            else:
                x_center = output[1] - CENTER_TO_LANE_PIXELS
                x_yaw = output[3]
        else:
            selected_lane = None

        #=========================================
        # 핸들 조향각 값인 angle값 정하기.
        # 차선의 위치 정보를 이용해서 angle값을 설정함.
        #=========================================
        # 칼만 필터로 상태 변수 추정 [0]=횡방향위치/방향 각도, [1]횡방향속도/방향 각속도
        xh_center, _ = kf_center.update(x_center)
        xh_yaw, _  = kf_yaw.update(x_yaw)

        # PID 제어를 통해 횡방향 제어용 조향각, 진행방향 제어용 조향각 계산
        angle_center = -int(pid_center.do(ref_center, xh_center[0], xh_center[1]))     
        angle_yaw = -int(pid_yaw.do(ref_yaw, xh_yaw[0], xh_yaw[1]))
        # 두 조향각을 합쳐서 하나의 핸들 조향각 결정
        angle = angle_center + angle_yaw

        #=========================================
        # 차량의 속도 값인 speed값 정하기.
        # 주행 속도를 조절하기 위해 speed값을 설정함.
        #=========================================
        # 차선이 하나라도 보이면 35의 속도, 그렇지 않으면 5의 속도로 이동
        # For debugging purpose speed is disabled! 

        #speed = 0
        speed = 0 if selected_lane is None else 5

        # img를 화면에 출력한다. 
        # For using this code in car display function temporary disabled!

        # text = 'xc=%d, xy=%d, ac(%d)+ay(%d)=angle(%d)'%(xh_center[0], xh_yaw[0], angle_center, angle_yaw, angle)
        # cv2.putText(img, text, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)
        # cv2.imshow("ICELAP", img)
        # cv2.waitKey(1)

        # 주행 속도를 결정
        # drive() 호출. drive()함수 안에서 모터 토픽이 발행됨.
        drive(angle, speed)
        
        rate.sleep()


#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함
# start() 함수가 실질적인 메인 함수임.
#=============================================
if __name__ == '__main__':
    start()

