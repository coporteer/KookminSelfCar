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

            near_id = rospy.loginfo(marker_lst[nearest_index]['id'])
            near_x = rospy.loginfo(marker_lst[nearest_index]['x'])
            near_z = rospy.loginfo(marker_lst[nearest_index]['z'])

    def usbcam_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def drive(self, angle, speed):        
        self.motor_msg.angle = angle
        self.motor_msg.speed = speed
        self.motor.publish(self.motor_msg)

    def near_drive_stop(self, z):

        Offset = 2
        # rospy.loginfo(z)

        if z <= Offset and z > 0:
            
            self.drive(0,0)
            rospy.loginfo(z)

        elif z > 0 :

            self.drive(0,7) 

        else :

            # rospy.loginfo("NoArDectected")

    def main_loop(self):
        rospy.loginfo('Waiting --------------')
        rospy.wait_for_message("/usb_cam/image_raw", Image)
        rospy.loginfo("Camera Ready --------------")

        self.xh_center=np.array([0,0])
        driving_mode = AutonomousDrive.DRIVING_MODE_STOP

        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            if driving_mode == AutonomousDrive.DRIVING_MODE_LANE:
                self.lane_drive(debug=True)

            self.near_drive_stop(self.near_z)

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


def signal_handler(sig, frame):
    os.system('killall -9 roslaunch roscore python')
    sys.exit(0)

if __name__ == '__main__':    
    signal.signal(signal.SIGINT, signal_handler)
    ads = AutonomousDrive()
    ads.main_loop()