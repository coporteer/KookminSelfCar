#!/usr/bin/env python3
# -*- coding: utf-8 -*- 16
from sensor_msgs.msg import Image
import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np

import signal
import sys
import os

class AutonomousDrive():

    def __init__(self):
        # CamThrower 코드시작, usb 카메라 토픽 구독 후 callback으로 넘김!
        rospy.init_node('CamThrower')
        rospy.Subscriber("/usb_cam/image_raw",Image, self.usbcam_callback, queue_size=1)

        ## 이미지 데이터를 저장하기위한 변수
        self.bridge = CvBridge()
        self.image = np.empty(shape=[0])

    def usbcam_callback(self, msg):
        # Callback에서는 ROS에서 넘어온 메세지를 이미지로 재구축!
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def main_loop(self):
        # 카메라(usb_cam/image_raw) 토픽 대기
        rospy.loginfo('Waiting --------------')
        rospy.wait_for_message("/usb_cam/image_raw", Image)
        rospy.loginfo("Camera Ready --------------")
        
        self.loop_hz = 30

        rate = rospy.Rate(self.loop_hz)

        while not rospy.is_shutdown():

            # 연속적인 이미지를 보여줍니다!
            cv2.imshow("ImageTest",self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        rate.sleep()

# 빠른 Ctrl+C를 위한 코드
def signal_handler(sig, frame):
    os.system('killall -9 roslaunch roscore python')
    sys.exit(0)

# 실행코드! 추가할 코드는 반드시 main_loop로!!
if __name__ == '__main__':    
    signal.signal(signal.SIGINT, signal_handler)
    ads = AutonomousDrive()
    ads.main_loop()