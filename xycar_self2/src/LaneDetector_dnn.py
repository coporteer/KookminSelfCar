#!/usr/bin/env python3
# -*- coding: utf-8 -*- 16
import cv2
import numpy as np
import math
import torch
import PIL
import torchvision.transforms as transforms

import rospy

class LaneDetector():
    NO_LINE_DETECTED = -1
    NO_VALID_LINE_DETECTED = -2
    FOUND_LANES = 1

    def __init__(self, width: int, height: int, canny_th1=60, canny_th2=75):
        self.WIDTH, self.HEIGHT = width, height
        self.ROI_START_ROW = 300    # roi의 y 시작
        self.ROI_END_ROW = 600      # roi의 y 끝
        self.MARGIN = 200           # 중앙 기준 차선을 찾지 않을 가로 px 너비의 반
        self.ROI_HEIGHT = self.ROI_END_ROW - self.ROI_START_ROW  # roi의 세로 길이
        self.L_ROW = self.ROI_HEIGHT-100             # 차선의 위치를 찾기 위한 ROI 안에서의 기준 Row값 
        self.LANE_INTERVAL = 660  # 차선 간격

        self.CANNY_TH1, self.CANNY_TH2 = canny_th1, canny_th2

        self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True).to('cuda')
        self.tf = transforms.ToTensor()

    def pre_process(self, img):
        img2 = self.tf(cv2.resize(img, (640, 640))).unsqueeze(dim=0)
        _, _,ll_seg_out = self.model(img2.to('cuda'))
        img = cv2.resize(ll_seg_out.to('cpu').detach().numpy().squeeze()[1,:,:], (self.WIDTH, self.HEIGHT))

        roi_img = img[self.ROI_START_ROW:self.ROI_END_ROW, 0:self.WIDTH]
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray = roi_img.copy()
        blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)*255
        edge_img = cv2.Canny(np.uint8(blur_gray), self.CANNY_TH1, self.CANNY_TH2)
        cv2.imshow('edge', edge_img)
        return roi_img.copy(), edge_img

    def do(self, img, display_img = False):
        # 1. canny edge
        roi_img, edge_img = self.pre_process(img)

        # 2. Hough transfom
        all_lines = cv2.HoughLinesP(edge_img, 1, math.pi/180,30,30,20)    
        if all_lines is None:
            return LaneDetector.NO_LINE_DETECTED, roi_img, edge_img, 0,0,0
        
        # 3. determining left and right lanes
        left_slopes = []
        right_slopes = []
        left_filtered_lines = []
        right_filtered_lines = []
        left_b = []
        right_b = []
        for line in all_lines:
            x1, y1, x2, y2 = line[0]

            if (x2 == x1):
                continue
            else:
                slope = float(y2-y1) / float(x2-x1)
        
            if 0.2 < abs(slope):
                if x2 < self.WIDTH/2 - self.MARGIN:
                    left_slopes.append(slope)
                    left_filtered_lines.append(line[0].tolist())
                    left_b.append(abs(y2-slope*x2))
                elif x1 > self.WIDTH/2 + self.MARGIN:
                    right_slopes.append(slope)
                    right_filtered_lines.append(line[0].tolist())
                    right_b.append(abs(y2-slope*x2))

        # 4. removing outliers
        left_b, right_b = np.array(left_b), np.array(right_b)
        if len(left_b) != 0:
            left_filtered_lines = np.delete(left_filtered_lines, np.where( abs(left_b-left_b.mean())>100), axis=0)
        if len(right_b) != 0:
            right_filtered_lines = np.delete(right_filtered_lines, np.where( abs(right_b-right_b.mean())>100), axis=0)

        if len(left_filtered_lines) + len(right_filtered_lines) == 0:
            return LaneDetector.NO_VALID_LINE_DETECTED, roi_img, edge_img, 0,0,0      
        
        # 5. determining the center of the left lane
        l_size = len(left_filtered_lines)
        m_left, m_right = 0, 0
        if l_size != 0:
            x_sum, y_sum, m_sum = 0.0, 0.0, 0.0
            for line in left_filtered_lines:
                x1,y1, x2,y2 = line
                x_sum += x1 + x2
                y_sum += y1 + y2
                m_sum += float(y2-y1)/float(x2-x1)
                if display_img:
                    cv2.line(roi_img, (x1,y1), (x2,y2), (0,0,255), 2)
            x_avg = x_sum / (l_size*2)
            y_avg = y_sum / (l_size*2)
            m_left = m_sum / l_size
            b_left = y_avg - m_left * x_avg
            if m_left != 0.0:
                x1 = int((0.0 - b_left) / m_left)
                x2 = int((self.ROI_HEIGHT - b_left) / m_left)
                if display_img:
                    cv2.line(roi_img, (x1,0), (x2,self.ROI_HEIGHT), (0,255,0), 2)
                x_left = int((self.L_ROW - b_left) / m_left)

        # 6. determining the center of the right lane
        r_size = len(right_filtered_lines)
        if r_size != 0:
            x_sum, y_sum, m_sum = 0.0, 0.0, 0.0
            for line in right_filtered_lines:
                x1,y1, x2,y2 = line
                x_sum += x1 + x2
                y_sum += y1 + y2
                m_sum += float(y2-y1)/float(x2-x1)
                if display_img:
                    cv2.line(roi_img, (x1,y1), (x2,y2), (0,0,255), 2)       
            x_avg = x_sum / (r_size*2)
            y_avg = y_sum / (r_size*2)
            m_right = m_sum / r_size
            b_right = y_avg - m_right * x_avg
            if m_right != 0.0:
                x1 = int((0.0 - b_right) / m_right)
                x2 = int((self.ROI_HEIGHT - b_right) / m_right)
                if display_img:
                    cv2.line(roi_img, (x1,0), (x2,self.ROI_HEIGHT), (255,0,255), 2)
                x_right = int((self.L_ROW - b_right) / m_right)

        if m_left == 0.0 and m_right != 0.0:            
            x_left = x_right - self.LANE_INTERVAL

        if m_left != 0.0 and m_right == 0.0:
            x_right = x_left + self.LANE_INTERVAL

        x_center = int((x_left+x_right)*0.5)
        return LaneDetector.FOUND_LANES, roi_img, edge_img, x_left, x_right, x_center
        
        
    def binarization(self, img):
        threshold = 160
        ### HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        _, _, v = cv2.split(hsv)
        hsv_low_white = np.array((0, 0, threshold))
        hsv_high_white = np.array((255, 255, 255))
        hsv_binary = cv2.inRange(hsv, hsv_low_white, hsv_high_white)

        ### HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        _, l, _ = cv2.split(hls)
        hls_low_white = np.array((0, threshold,  0))
        hls_high_white = np.array((255, 255, 255))
        hls_binary = cv2.inRange(hls, hls_low_white, hls_high_white)

        ### R color channel 경계값 설정
        _, _, r = cv2.split(img)
        r_low_white = threshold
        r_high_white = 255
        r_binary = cv2.inRange(r, r_low_white, r_high_white)        
        combined = np.asarray(r_binary/255, dtype=np.uint8) +  np.asarray(hls_binary/255, dtype=np.uint8) + np.asarray(hsv_binary/255, dtype=np.uint8)
        
        combined[combined < 2] = 0
        combined[combined >= 2] = 255
        return  combined
    