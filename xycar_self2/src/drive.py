#!/usr/bin/env python3
# -*- coding: utf-8 -*- 16
from sensor_msgs.msg import Image
from xycar_msgs.msg import xycar_motor
from ar_track_alvar_msgs.msg import AlvarMarkers

import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np

import signal
import sys
import os

from LaneDetector import LaneDetector
from Kalman import KalmanPos2Vel
from PID import PID

class AutonomousDrive():
    DRIVING_MODE_STOP = -1
    DRIVING_MODE_LANE = 0

    global near_x,near_y,near_z,near_id
    global OutputE, Error, Steer

    near_x = 0
    near_y = 0
    near_z = 0
    near_id = 0
    OutputE = 0
    Error = 0
    Steer = 0

    def __init__(self):
        rospy.init_node('xycar_self')
        rospy.Subscriber("/usb_cam/image_raw",Image, self.usbcam_callback, queue_size=1)
        rospy.Subscriber('/ar_pose_marker/', AlvarMarkers, self.ar_callback, queue_size=1)
        self.motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
        self.motor_msg = xycar_motor()

        self.loop_hz = 30
        camera_width, camera_height = 1280, 720
        
        ## lane drive 관련
        self.ref_center = camera_width//2
        self.ld = LaneDetector(width=camera_width, height=camera_height, canny_th1=30, canny_th2=60)
        self.kf_lane = KalmanPos2Vel(x0=np.array([camera_width//2,0]), P0=5*np.eye(2), q1=0.5, q2=1, r=20, dt=1/self.loop_hz)
        self.pid_lane = PID(Kp=1.5, Ki=0.0, Kd=0.1, dt=1/self.loop_hz, max_u=40)
        self.speed_lane_found = 20
        self.speed_lane_not_found = 10

        ## 

        self.bridge = CvBridge()
        self.image = np.empty(shape=[0])

    def ar_callback(self, msg):
        # AR 주행 시 id 순서
        # 3 6 8 5 4 7
        marker_lst = []
        dist_list = []
        if len(msg.markers) != 0:
            for marker in msg.markers:
                marker_info = dict()
                marker_info['id'] = marker.id
                marker_info['x'] = marker.pose.pose.position.x
                marker_info['y'] = marker.pose.pose.position.y
                marker_info['z'] = marker.pose.pose.position.z
                dist_list.append(abs(marker.pose.pose.position.x)+abs(marker.pose.pose.position.y)+abs(marker.pose.pose.position.z))
                marker_lst.append(marker_info)
            nearest_index = dist_list.index(min(dist_list))
            # rospy.loginfo(f"{marker_lst[nearest_index]['id']}, ({marker_lst[nearest_index]['x']:0.2f}, {marker_lst[nearest_index]['y']:0.2f}, {marker_lst[nearest_index]['z']:0.2f})")
            # rospy.loginfo(marker_lst)

            global near_x,near_y,near_z,near_id

            near_x = marker_lst[nearest_index]['x']
            near_id = marker_lst[nearest_index]['id']
            near_z = marker_lst[nearest_index]['z']
            near_y = marker_lst[nearest_index]['y']

    def usbcam_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def drive(self, angle, speed):        
        self.motor_msg.angle = angle
        self.motor_msg.speed = speed
        self.motor.publish(self.motor_msg)

    def main_loop(self):

        global Error
        global Steer

        rospy.loginfo('Waiting --------------')
        rospy.wait_for_message("/usb_cam/image_raw", Image)
        rospy.loginfo("Camera Ready --------------")

        self.xh_center=np.array([0,0])
        driving_mode = AutonomousDrive.DRIVING_MODE_STOP

        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            if driving_mode == AutonomousDrive.DRIVING_MODE_LANE:
                self.lane_drive(debug=True)

        
            # rospy.loginfo(near_x)
            Error = self.theta_cal(near_x,near_z) * 2
            # rospy.loginfo(Error)
            # rospy.loginfo(f'{near_x:0.3f},{near_z:0.3f},{near_y:0.3f}')
            # Steer = self.NewDrive(near_x,near_z)
            Steer = self.SpeedPID(near_z)
            self.near_drive_stop(Error,Steer)
            # self.drive(Error,Steer)
            rate.sleep()

    def lane_drive(self, debug=False):
        ret, roi_img, edge_img, x_left, x_right, x_center = self.ld.do(self.image, debug)
        if ret==LaneDetector.FOUND_LANES:
            self.xh_center, _ = self.kf_lane.update(x_center)

        u = self.pid_lane.do(self.ref_center, self.xh_center[0], self.xh_center[1])
        rospy.loginfo(f'x={self.xh_center[0]:0.1f}, r={self.ref_center}, u={int(u)}')
        self.drive(int(u), self.speed_lane_found if ret==LaneDetector.FOUND_LANES else self.speed_lane_not_found)

        if debug:
            cv2.rectangle(roi_img, (x_left-5,self.ld.L_ROW-5), (x_left+5,self.ld.L_ROW+5), (0,255,0), 4)
            cv2.rectangle(roi_img, (x_right-5,self.ld.L_ROW-5), (x_right+5,self.ld.L_ROW+5), (255,0,255), 4)        
            cv2.rectangle(roi_img, (int(self.xh_center[0])-5,self.ld.L_ROW-5), (int(self.xh_center[0])+5,self.ld.L_ROW+5), (255,0,0), 4)
            cv2.imshow('roi_img', roi_img)
            cv2.imshow('edge_img', edge_img)
            cv2.waitKey(1)
    
    def near_drive_stop(self, e, z):

        Offset = 0.15
        # rospy.loginfo(z)

        if z <= Offset and z > 0:

            self.drive(0,0)
            rospy.loginfo("stop")

        elif z > Offset and z > 0:

            self.drive(e,z)
            rospy.loginfo("drive")

        else :

            rospy.loginfo("ParameterError!")

    def theta_cal(self, x, z):
        
        # So ARcode coordinates are (x, z)
        RangeOffset = 0.05
        # inilist = [0,0]
        # theta = np.array(inilist)
        x = x - 0.01

        # Target coordinates are (x, z - 0.15)
        TargetPointZ = z - RangeOffset

        theta = np.arctan(TargetPointZ/x) * 180/np.pi

        if theta < 0:
            theta = 180 + theta
        #rospy.loginfo(theta[1])

        theta_error = 90 - theta

        #rospy.loginfo(theta)

        return theta_error

    # def SteeringE(self, InputE):
        
    #     # e value range is 90~135(right) 45~90(left)
    #     # so i will range those values tnto xycarmotor 0~50(right) -50~0(left)
    
    #     InputEMax = 50
    #     InputE0 = 0
    #     InputEmin = -50

    #     OutputEMax = 20
    #     OutputE0 = 0
    #     OutputEMin = -20

    #     global OutputE 


    #     # Using scale conversion!
    #     # new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    #     if InputE0 < InputE < InputEMax + 1 :

    #         OutputE = (InputE - InputE0) / (InputEMax - InputE0) * (OutputEMax - OutputE0) + OutputE0

    #     elif InputEmin - 1 < InputE < InputE0 :

    #         OutputE = (InputE - InputEmin) / (InputE0 - InputEmin) * (OutputE0 - OutputEMin) + OutputEMin

    #     elif InputE == InputE0 :

    #         OutputE = OutputE0

    #     else :

    #         rospy.loginfo("ConversionError")

    
    #     # rospy.loginfo(OutputE)

    #     return OutputE

    def SpeedPID(self, z):

        # 이제 AR 태그와 Z거리를 이용해 속도를 제어해보자
            # z값은 약 0.01에서 10까지도 나오기에 P를 100으로 놓고 최고 속도는 10으로 생각했다
            # dt 값은 변화량이므로 그대로 놓았다.
            pid_speed = PID(Kp=100, Ki=0.00, Kd=0.00, dt=0.01, max_u =10)
            speed = -int(pid_speed.do(0.15, z, 0))

            return speed

def signal_handler(sig, frame):
    os.system('killall -9 roslaunch roscore python')
    sys.exit(0)

if __name__ == '__main__':    
    signal.signal(signal.SIGINT, signal_handler)
    ads = AutonomousDrive()
    ads.main_loop()