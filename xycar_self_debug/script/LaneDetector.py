# -*- coding: utf-8 -*-
"""
이 파일은 과제 2의 자율 주행을 위한 차선 인식 코드임
차선 인식을 위해 다양한 영상처리 기법을 사용하여 차선을 인식함
"""

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import cv2
import numpy as np

class LaneDetector():
    '''차선 인식 클래스
    '''
    ERR_NO_WHITE_PIXELS = 0
    ERR_HISTOGRAM_NOT_VALID = 1
    ERR_NOT_VALID = 2
    ERR_NOT_FOUND = 3
    SUCCESS_POLY_FIT_ONE = -1
    SUCCESS_POLY_FIT_BOTH = -2

    def __init__(self, 
                 img_shape,                 # img size
                 pt_src,                    # perspective transformation (src)
                 pt_dst,                    # perspective transformation (dst)
                 roi_vertices,              # ROI crop coordinate
                 nb_windows = 24,           # number of sliding windows
                 margin = 50,               # width of the windows +/- margin
                 minpix = 50,               # min number of pixels needed to recenter the window
                 min_lane_pts = 500):       # min number of 'hot' pixels needed to fit a 2nd order polynomial as a lane line
        self.img_width, self.img_height = img_shape
        self.pt_src = pt_src
        self.pt_dst = pt_dst
        self.nb_windows = nb_windows
        self.margin = margin
        self.minpix = minpix
        self.min_lane_pts = min_lane_pts
        self.window_height = int(self.img_height / self.nb_windows)
        self.vertices = roi_vertices
        self.crop_height = 300
        self.offset = 30

    def do(self, img):
        """ 차선 인식 알고리즘을 수행

        Args:
            img (numpy.array): 원본 입력 이미지
        
        Return:
            ret (int): 인식한 차선의 상태를 나타내는 변수
            img (numpy.array): 원본 입력 이미지 + sliding window 결과 이미지
            output (list): 인식된 차선의 정보 [왼쪽 차선의 중앙 pixel, 오른쪽 차선의 중앙 pixel, 왼쪽 차선의 기울어진 각도, 오른쪽 차선의 기울어진 각도]
        """

        roi_img = self.preprocess(img)  # 원본 이미지 탑-다운 뷰로 변환하는 전처리 과정
        binary_img = self.binarization(roi_img) # 탑-다운뷰 이미지 이진화
        ret, out_img, left_fit, right_fit, left_angle, right_angle = self.polyfit(binary_img) # 이진화 이미지에서 차선 인식

        output = [-1,-1, -1, -1]
        if ret <= LaneDetector.SUCCESS_POLY_FIT_ONE:
            # 하나 이상 차선을 찾은 경우
            if left_fit is not None:
                # 왼쪽 차선을 찾은 경우 왼쪽 차선의 중앙 pixel, 왼쪽 차선의 기울어진 각도 계산
                output[0] = self.calc_output(left_fit)
                output[2] = left_angle
            if right_fit is not None:
                # 오른쪽 차선을 찾은 경우 오른쪽 차선의 중앙 pixel, 오른쪽 차선의 기울어진 각도 계산
                output[1] = self.calc_output(right_fit)
                output[3] = right_angle

            # 원본 이미지에 찾은 차선의 2차 곡선 그려 넣기
            img = self.draw(img, roi_img, self.invM, left_fit, right_fit)

        # 두 개의 차선을 찾았는데, 두 차선이 서로 다른 방향을 바라보면 잘못된 정보라고 판단
        if ret==LaneDetector.SUCCESS_POLY_FIT_BOTH and abs(output[2]-output[3]) > 5:
            ret = LaneDetector.ERR_NOT_VALID
            
        # 원본 이미지와 sliding window 결과 이미지를 합침
        img = np.hstack([img[self.img_height-self.img_height:self.img_height, 0:self.img_width], out_img]) 
        return ret, img, output
    
    def preprocess(self, img):
        """ 이미지 전처리 단계: 원하는 영역의 이미지를 탑다운 뷰로 변환

        Args:
            img (numpy.array): 윈본 입력 이미지

        Return:
            roi (numpy.array): 탑다운뷰로 변환한 관심영역
        """

        # 1. Perspective transformation
        self.M = cv2.getPerspectiveTransform(self.pt_src, self.pt_dst) # pt_src좌표에서 pt_dst좌표로 대응되는 원근 변환 행렬(탑-다운뷰료 변환)
        self.invM = cv2.getPerspectiveTransform(self.pt_dst, self.pt_src) # pt_dst좌표에서 pt_src좌표로 대응되는 원근 변환 행렬(탑-다운뷰에서 원본 이미지로 변환)
        warped = cv2.warpPerspective(img, self.M, (self.img_width, self.img_height), flags=cv2.INTER_LINEAR) # 원본 이미지를 탑-다운뷰 이미지로 변환

        # 2. ROI crop    
        if len(warped.shape) == 3:
            fill_color = (255,) * 3
        else:
            fill_color = 255
                
        mask = np.zeros_like(warped) # warped와 같은 크기의 0 행렬 생성
        mask = cv2.fillPoly(mask, self.vertices, fill_color) # vertices 좌표에 fill_color를 가지는 다각형의 mask 생성  
        roi = cv2.bitwise_and(warped, mask) # 비트연산을 통해 warped 이미지에 mask 적용

        #cv2.imshow("roi",roi)

        return roi
    
    def binarization(self, img):
        """ 이미지 이진화 처리: 여러 영상처리 기법을 사용하여 이미지를 바이너리 형태로 생성 

        Args:
            img (numpy.array): 윈본 입력 이미지

        Return:
            combined (numpy.array): 이진화된 이미지
        """

        OffsetA = 0.925
        OffsetR = 0.785
        OffsetHSV = 0.815
        OffestHSl = 0.815

        ### HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # RGB이미지를 HSV영역으로 불러옴
        V = hsv[:,:,2]
        V_max, V_mean = np.max(V), np.mean(V)
        
        # HSV_WHITE 경계값 설정
        V_adapt_white = max(150, int(V_max * 0.8),int(V_mean * 1.25))
        hsv_low_white = np.array((0, 0, OffsetHSV * V_adapt_white))
        hsv_high_white = np.array((255, 40, 220))

        hsv_binary = self.binary_threshold(hsv, hsv_low_white, hsv_high_white) # hsv_white 경계값 적용

        ### HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) # RGB이미지를 HLS영역으로 불러옴
        L = hls[:,:,1]
        L_max, L_mean = np.max(L), np.mean(L)
    
        # HLS_WHITE 경계값 설정
        L_adapt_white =  max(225, int(L_max *0.8),int(L_mean * 1.25))  # increased to 225
        hls_low_white = np.array((0, OffestHSl * L_adapt_white,  0))
        hls_high_white = np.array((255, 255, 255))

        hls_binary = self.binary_threshold(hls, hls_low_white, hls_high_white) # hls_white 경계값 적용

        ### R color channel 경계값 설정
        R = img[:,:,0]
        R_max, R_mean = np.max(R), np.mean(R)
        
        R_low_white = min(max(150, int(R_max * 0.55), int(R_mean * 1.95)),230)
        R_binary = self.binary_threshold(R, OffsetR * R_low_white, 255) # R color 경계값 적용
        
        ### Ensemble Voting
        combined = np.asarray(R_binary +  hls_binary + hsv_binary, dtype=np.uint8) # HSV, HLS, R color threshold를 적용한 결과 합치기

        combined[combined < OffsetA] = 0
        combined[combined >= OffsetA] = 1

        return  combined

    def binary_threshold(self, img, low, high):
        """ 이진화 이미지에 경계값 적용 : 바이너리 이미지에 경계값을 이용하여 마스크를 씌움

        Args:
            img (numpy.array): 윈본 입력 이미지
            low (numpy.array): white 경계값의 하한치
            high (numpy.array): white 경계값의 상한치

        Return:
            output (numpy.array): 경계값 적용을 한 이진화 이미지
        """

        # 이미지의 차원과 경계값에 따른 마스트 생성
        if len(img.shape) == 2:
            output = np.zeros_like(img)
            mask = (img >= low) & (img <= high)
            
        elif len(img.shape) == 3:
            output = np.zeros_like(img[:,:,0])
            mask = (img[:,:,0] >= low[0]) & (img[:,:,0] <= high[0]) \
                & (img[:,:,1] >= low[1]) & (img[:,:,1] <= high[1]) \
                & (img[:,:,2] >= low[2]) & (img[:,:,2] <= high[2])
                
        output[mask] = 1
        return output
    
    def polyfit(self, binary):
        """ 이미지에서 차선을 인식: 바이너리 이미지에서 상황별 차선을 인식

        Args:
            binary (numpy.array): 윈본 바이너리 입력 이미지

        Return:
            ret (int): 차선 인식 갯수에 따른 상태를 나타내는 변수
            out (numpy.array): 차선을 인식한 출력 이미지
            left_fit (numpy.array): 왼쪽 차선의 2차 피팅 정보
            right_fit (numpy.array): 오른쪽 차선의 2차 피팅 정보
            left_angle (numpy.float64): 왼쪽 차선의 기울기 각도
            right_angle (numpy.float64): 오른쪽 차선의 기울기 각도
        """

        # 이진 영상을 3차원 R,G,B 영상으로 변환
        out = np.dstack((binary, binary, binary)) * 255

        if binary.max() <= 0: # no white pixels
            return LaneDetector.ERR_NO_WHITE_PIXELS, out, None, None, None, None
        
        # 세로 방향으로 히스토그램 계산
        histogram = np.sum(binary[self.crop_height:,:], axis=0)
        if histogram.max() <= 0:
            return LaneDetector.ERR_HISTOGRAM_NOT_VALID, out, None, None, None, None
        
        # 영상의 가로를 반으로 나누어서 왼쪽과 오른쪽 영역에서 히스토그램이 가장 높은 위치를 차선 찾는 기준 위치로 선택
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint-self.offset])
        rightx_base = np.argmax(histogram[midpoint+self.offset:]) + midpoint

        nonzero = binary.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        # sliding window를 만들어서 위로 올라가며 차선의 후보들을 검색
        for window in range(self.nb_windows):
            win_y_low = self.img_height - (1 + window) * self.window_height
            win_y_high = self.img_height - window * self.window_height

            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin

            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Sliding window를 그린다.
            cv2.rectangle(out, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),\
                        (0, 255, 0), 2)
            cv2.rectangle(out, (win_xright_low, win_y_low), (win_xright_high, win_y_high),\
                        (0, 255, 0), 2)

            # Sliding window에 포함되는 흰 pixel들의 좌표 index를 저장
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
                            & (nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
                            & (nonzerox >= win_xright_low) & (nonzerox <= win_xright_high)).nonzero()[0]

            # Sliding window에 포함되는 흰 pixel의 수가 최소 기준 이상이면 이것들을 차선에 대한 pixel로 포함
            if len(good_left_inds) >  self.minpix:
                left_lane_inds.append(good_left_inds)
                leftx_current = int(np.mean(nonzerox[good_left_inds]))

            if len(good_right_inds) > self.minpix:
                right_lane_inds.append(good_right_inds)
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_fit, right_fit = None, None
        left_angle, right_angle = (0,0)
        leftx, rightx = [], []

        # 왼쪽과 오른쪽 차선에 대한 pixel들의 x,y 좌표 목록을 만든다.
        if left_lane_inds != []:
            left_lane_inds = np.concatenate(left_lane_inds)
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]

        if right_lane_inds != []:
            right_lane_inds = np.concatenate(right_lane_inds)
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

        # 찾은 차선 후보 pixel의 총 수가 기준값 이상이고, 왼쪽과 오른쪽 차선의 x좌표 평균이 각각 화면의 왼쪽과 오른쪽에 있다면 차선 후보로 선택
        # 선택된 좌표들에 대해 2차 함수 fitting 수행
        if len(leftx) >= self.min_lane_pts and np.mean(leftx) < self.img_width//2-self.offset:
            left_fit = np.polyfit(lefty, leftx, 2)
            out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            left_angle = self.get_angle(left_fit)
        if len(rightx) >= self.min_lane_pts and np.mean(rightx) > self.img_width//2+self.offset:
            right_fit = np.polyfit(righty, rightx, 2)
            out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 255]
            right_angle= self.get_angle(right_fit)
        
        if left_fit is not None and right_fit is not None:
            # 양쪽 차선에 대한 차선 후보를 모두 찾으면 유효한지 검사
            if self.check_validity(left_fit, right_fit):
                # 찾은 두 개의 차선이 유효
                ret = LaneDetector.SUCCESS_POLY_FIT_BOTH
                return ret, out, left_fit, right_fit, left_angle, right_angle
            else:
                # 그렇지 않음
                ret = LaneDetector.ERR_NOT_VALID
                return ret, out, None, None, None, None
        elif left_fit is not None or right_fit is not None:
            # 둘 중 하나의 차선만 찾은 경우 유효성 검사를 하지 않고 이대로 결정
            ret = LaneDetector.SUCCESS_POLY_FIT_ONE
            return ret, out, left_fit, right_fit, left_angle, right_angle
        else:
            # 아무 차선도 찾지 못함
            ret = LaneDetector.ERR_NOT_FOUND
            return ret, out, None, None, None, None

    def get_angle(self, fit):
        """ 차선의 기울기 값을 계산: 각 차선이 차량의 진행 방향과 이루는 각도를 계산

        Args:
            fit (numpy.array): 인식된 차선을 2차 피팅한 정보

        Return:
            angle (numpy.float64): 인식된 차선의 기울기 값
        """

        # 차선의 2차 곡선의 제일 아래와 길이의 75% 위치에서 두 점을 선택해서 두 점이 이루는 각도 계산
        _, poly_y = self.get_poly_points(fit)
        y1 = self.img_height - 1 # Bottom
        y2 = self.img_height - int(len(poly_y)* 0.75)
        x1 = fit[0]  * (y1**2) + fit[1]  * y1 + fit[2]
        x2 = fit[0]  * (y2**2) + fit[1]  * y2 + fit[2]

        angle = np.rad2deg(np.arctan2(y2-y1, x2-x1)) # 라디안을 degree로 변환

        return angle

    def calc_output(self, fit):
        """ 차선 x 좌표의 중앙 값 반환: 차선 좌표에서 중앙에 위치하는 x 좌표 값

        Args:
            fit (numpy.array): 인식된 차선의 2차 피팅 정보

        Return:
            result (numpy.int64): 인식된 차선의 x좌표의 중앙 값
        """
        if fit is not None:
            poly_x, poly_y = self.get_poly_points(fit)
            result = poly_x[len(poly_x)//2]

            return result

    def get_poly_points(self,fit):
        """ 인식된 차선의 2차 피팅 정보를 x,y 좌표로 변환

        Args:
            fit (numpy.array): 인식된 차선의 2차 피팅 정보

        Return:
            x (numpy.array): 인식된 차선의 x 좌표
            y (numpy.array): 인식되 차선의 y 좌표
        """

        ysize, xsize = self.img_height, self.img_width
        
        plot_y = np.linspace(0, ysize-1, ysize)
        plot_x = fit[0] * plot_y**2 + fit[1] * plot_y + fit[2]
        plot_x = plot_x[(plot_x >= 0) & (plot_x <= xsize - 1)]
        plot_y = np.linspace(ysize - len(plot_x), ysize - 1, len(plot_x))

        x = plot_x.astype(int)
        y = plot_y.astype(int)

        return x, y
   
    def draw(self,img, warped, invM, left_fit, right_fit):
        """ 원본 이미지에 찾은 차선의 2차 곡선 그려 넣기

        Args:
            fit (numpy.array): 인식된 차선의 2차 피팅 정보
            img (numpy.array): 원본 이미지
            warped (numpy.array): 탑-다운뷰 이미지
            invM (numpy.array): 탑-다운뷰 이미지를 원래 이미지로 변환하는 원근 변환 행렬
            left_fit (numpy.array): 왼쪽 차선의 2차 피팅 정보
            right_fit (numpy.array): 오른쪽 차선의 2차 피팅 정보

        Return:
            out (numpy.array): 최종 출력 이미지
        """

        warp_zero = np.zeros_like(warped[:,:,0]).astype(np.uint8) # warped 이미지 사이즈의 0 numpy.array
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero)) # warp_zero를 3차원의 numpy.array로 stack
        
        # 인식된 차선의 유무에 따라 차선을 그림
        if left_fit is not None:
            plot_xleft, plot_yleft = self.get_poly_points(left_fit)
            pts_left = np.array([np.transpose(np.vstack([plot_xleft, plot_yleft]))])
            cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False,
                            color=(255, 0, 0), thickness=10)
        if right_fit is not None:
            plot_xright, plot_yright = self.get_poly_points(right_fit)
            pts_right = np.array([np.flipud(np.transpose(np.vstack([plot_xright, plot_yright])))])
            cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False,
                        color=(255, 0, 255), thickness= 10)
 
        # 차선 정보를 포함하는 탑-다운뷰 이미지를 원래 이미지로 원근 변환함
        unwarped = cv2.warpPerspective(color_warp, invM, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR) 
        
        lane1=np.where((unwarped[:,:,0]!=0) & (unwarped[:,:,2]==0))
        lane2=np.where((unwarped[:,:,2]!=0) & (unwarped[:,:,0]!=0))
        out = img.copy()

        # 인식된 차선을 원본 이미지에 표시
        out[lane1]=(255,0,0)
        out[lane2]=(255,0,255)

        return out

    def check_validity(self, left_fit, right_fit):
        """ 왼쪽, 오른쪽 차선의 존재 유무 판단

        Args:
            left_fit (numpy.array): 인식된 왼쪽 차선의 2차 피팅 정보
            right_fit (numpy.array): 인식된 오른쪽 차선의 2차 피팅 정보

        Return:
            valid (bool): 왼쪽, 오른쪽 두 차선 모두 존재하면 True, 아니면 False
        """

        valid = True

        if left_fit is None or right_fit is None:
            return False, True, True

        _, poly_yleft = self.get_poly_points(left_fit)
        _, poly_yright = self.get_poly_points(right_fit)

        # 두 선이 서로 다른 세 개의 Y 값에 대해 서로 그럴듯한 거리 안에 있는지 확인
        y1 = self.img_height - 1 # Bottom
        y2 = self.img_height - int(min(len(poly_yleft), len(poly_yright)) * 0.35) # 두 번째와 세 번째의 경우 y1과 사용 가능한 최상위 값 사이의 값을 가져옴
        y3 = self.img_height - int(min(len(poly_yleft), len(poly_yright)) * 0.75)

        # 두 라인의 x값을 계산
        x1l = left_fit[0]  * (y1**2) + left_fit[1]  * y1 + left_fit[2]
        x2l = left_fit[0]  * (y2**2) + left_fit[1]  * y2 + left_fit[2]
        x3l = left_fit[0]  * (y3**2) + left_fit[1]  * y3 + left_fit[2]

        x1r = right_fit[0] * (y1**2) + right_fit[1] * y1 + right_fit[2]
        x2r = right_fit[0] * (y2**2) + right_fit[1] * y2 + right_fit[2]
        x3r = right_fit[0] * (y3**2) + right_fit[1] * y3 + right_fit[2]

        # L1 norm 계산
        x1_diff = abs(x1l - x1r)
        x2_diff = abs(x2l - x2r)
        x3_diff = abs(x3l - x3r)

        # 3점에 대한 임계값 설정
        min_dist_y1 = 300 
        max_dist_y1 = 600 
        min_dist_y2 = 250 
        max_dist_y2 = 600 
        min_dist_y3 = 200 
        max_dist_y3 = 600 
        
        if (x1_diff < min_dist_y1) | (x1_diff > max_dist_y1) | \
            (x2_diff < min_dist_y2) | (x2_diff > max_dist_y2) | \
            (x3_diff < min_dist_y3) | (x3_diff > max_dist_y3):
            valid = False

        # 두 개의 서로 다른 Y 값에 대해 선 기울기가 유사한지 확인
        # x = Ay**2 + By + C
        # dx/dy = 2Ay + B
        y1left_dx  = 2 * left_fit[0]  * y1 + left_fit[1]
        y3left_dx  = 2 * left_fit[0]  * y3 + left_fit[1]
        y1right_dx = 2 * right_fit[0] * y1 + right_fit[1]
        y3right_dx = 2 * right_fit[0] * y3 + right_fit[1]

        # L1 norm 계산
        norm1 = abs(y1left_dx - y1right_dx)
        norm2 = abs(y3left_dx - y3right_dx)

        # L1 norm 임계값 설정
        thresh = 0.6 #0.58 
        if (norm1 >= thresh) | (norm2 >= thresh):
                valid = False
       
        return valid
