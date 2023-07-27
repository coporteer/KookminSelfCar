#!/usr/bin/env python3
# -*- coding: utf-8 -*- 16

import numpy as np
import cv2, rospy, time, math
from sensor_msgs.msg import Image
from xycar_msgs.msg import xycar_motor
from cv_bridge import CvBridge


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

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
bridge = CvBridge()  # OpenCV 함수를 사용하기 위한 브릿지 
image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
WIDTH, HEIGHT = 1280, 720  # 카메라 이미지 가로x세로 크기
Blue = (0,255,0) # 파란색
Green = (0,255,0) # 녹색
Red = (0,0,255) # 빨간색
Yellow = (0,255,255) # 노란색
View_Center = WIDTH//2  # 화면의 중앙값 = 카메라 위치

#=============================================
# 차선 인식 프로그램에서 사용할 상수 선언부
#=============================================
CAM_FPS = 30  # 카메라 FPS 초당 30장의 사진을 보냄
WIDTH, HEIGHT = 1280, 720  # 카메라 이미지 가로x세로 크기
ROI_START_ROW = 400  # 차선을 찾을 ROI 영역의 시작 Row값
ROI_END_ROW = 720  # 차선을 찾을 ROT 영역의 끝 Row값
ROI_HEIGHT = ROI_END_ROW - ROI_START_ROW  # ROI 영역의 세로 크기  <- roi area!
L_ROW = 40  # 차선의 위치를 찾기 위한 ROI 안에서의 기준 Row값 

#=============================================
# 콜백함수 - USB 전방카메라 토픽을 처리하는 콜백함수.
#=============================================
def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

#=============================================
# 카메라 영상 이미지에서 차선을 찾아 그 위치를 반환하는 코드
#=============================================
def lane_detect():
    global image
    prev_x_left = 0
    prev_x_right = WIDTH

    img = image.copy() # 이미지처리를 위한 카메라 원본이미지 저장
    display_img = img  # 디버깅을 위한 디스플레이용 이미지 저장
    
    # img(원본이미지)의 특정영역(ROI Area)을 잘라내기
    roi_img = img[ROI_START_ROW:ROI_END_ROW, 0:WIDTH]
    line_draw_img = roi_img.copy()

    #=========================================
    # 원본 칼라이미지를 그레이 회색톤 이미지로 변환하고 
    # 블러링 처리를 통해 노이즈를 제거한 후에 (약간 뿌옇게, 부드럽게)
    # Canny 변환을 통해 외곽선 이미지로 만들기
    #=========================================
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)
    edge_img = cv2.Canny(np.uint8(blur_gray), 60, 75)

    #cv2.imshow("canny", edge_img)

    # 잘라낸 이미지에서 HoughLinesP 함수를 사용하여 선분들을 찾음
    all_lines = cv2.HoughLinesP(edge_img, 1, math.pi/180,50,50,20)
    
    if all_lines is None:
        return False, 0, 0

    #=========================================
    # 선분들의 기울기 값을 각각 모두 구한 후에 리스트에 담음. 
    # 기울기의 절대값이 너무 작은 경우 (수평선에 가까운 경우)
    # 해당 선분을 빼고 담음. 
    #=========================================
    slopes = []
    filtered_lines = []

    for line in all_lines:
        x1, y1, x2, y2 = line[0]

        if (x2 == x1):
            slope = 1000.0
        else:
            slope = float(y2-y1) / float(x2-x1)
    
        if 0.2 < abs(slope):
            slopes.append(slope)
            filtered_lines.append(line[0])

    if len(filtered_lines) == 0:
        return False, 0, 0

    #=========================================
    # 왼쪽 차선에 해당하는 선분과 오른쪽 차선에 해당하는 선분을 구분하여 
    # 각각 별도의 리스트에 담음.
    #=========================================
    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = filtered_lines[j]
        slope = slopes[j]

        x1,y1, x2,y2 = Line

        # 기울기 값이 음수이고 화면의 왼쪽에 있으면 왼쪽 차선으로 분류함
        # 기준이 되는 X좌표값 = (화면중심값 - Margin값)
        Margin = 0
        
        if (slope < 0) and (x2 < WIDTH/2-Margin):
            left_lines.append(Line.tolist())

        # 기울기 값이 양수이고 화면의 오른쪽에 있으면 오른쪽 차선으로 분류함
        # 기준이 되는 X좌표값 = (화면중심값 + Margin값)
        elif (slope > 0) and (x1 > WIDTH/2+Margin):
            right_lines.append(Line.tolist())

    # 디버깅을 위해 차선과 관련된 직선과 선분을 그리기 위한 도화지 준비
    line_draw_img = roi_img.copy()
    
    # 왼쪽 차선에 해당하는 선분은 빨간색으로 표시
    for line in left_lines:
        x1,y1, x2,y2 = line
        cv2.line(line_draw_img, (x1,y1), (x2,y2), Red, 2)

    # 오른쪽 차선에 해당하는 선분은 노란색으로 표시
    for line in right_lines:
        x1,y1, x2,y2 = line
        cv2.line(line_draw_img, (x1,y1), (x2,y2), Yellow, 2)

    #=========================================
    # 왼쪽/오른쪽 차선에 해당하는 선분들의 데이터를 적절히 처리해서 
    # 왼쪽차선의 대표직선과 오른쪽차선의 대표직선을 각각 구함.
    # 기울기와 Y절편값으로 표현되는 아래와 같은 직선의 방적식을 사용함.
    # (직선의 방정식) y = mx + b (m은 기울기, b는 Y절편)
    #=========================================

    # 왼쪽 차선을 표시하는 대표직선을 구함        
    m_left, b_left = 0.0, 0.0
    x_sum, y_sum, m_sum = 0.0, 0.0, 0.0

    # 왼쪽 차선을 표시하는 선분들의 기울기와 양끝점들의 평균값을 찾아 대표직선을 구함
    size = len(left_lines)
    if size != 0:
        for line in left_lines:
            x1, y1, x2, y2 = line
            x_sum += x1 + x2
            y_sum += y1 + y2
            if(x2 != x1):
                m_sum += float(y2-y1)/float(x2-x1)
            else:
                m_sum += 0                
            
        x_avg = x_sum / (size*2)
        y_avg = y_sum / (size*2)
        m_left = m_sum / size
        b_left = y_avg - m_left * x_avg

        if m_left != 0.0:
            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = 0
            # 위 두 직선의 교점의 좌표값 (x1, 0)을 구함.           
            x1 = int((0.0 - b_left) / m_left)

            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = ROI_HEIGHT
            # 위 두 직선의 교점의 좌표값 (x2, ROI_HEIGHT)을 구함.               
            x2 = int((ROI_HEIGHT - b_left) / m_left)

            # 두 교점, (x1,0)과 (x2, ROI_HEIGHT)를 잇는 선을 그림
            cv2.line(line_draw_img, (x1,0), (x2,ROI_HEIGHT), Blue, 2)

    # 오른쪽 차선을 표시하는 대표직선을 구함      
    m_right, b_right = 0.0, 0.0
    x_sum, y_sum, m_sum = 0.0, 0.0, 0.0

    # 오른쪽 차선을 표시하는 선분들의 기울기와 양끝점들의 평균값을 찾아 대표직선을 구함
    size = len(right_lines)
    if size != 0:
        for line in right_lines:
            x1, y1, x2, y2 = line
            x_sum += x1 + x2
            y_sum += y1 + y2
            if(x2 != x1):
                m_sum += float(y2-y1)/float(x2-x1)
            else:
                m_sum += 0     
       
        x_avg = x_sum / (size*2)
        y_avg = y_sum / (size*2)
        m_right = m_sum / size
        b_right = y_avg - m_right * x_avg

        if m_right != 0.0:
            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = 0
            # 위 두 직선의 교점의 좌표값 (x1, 0)을 구함.           
            x1 = int((0.0 - b_right) / m_right)

            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = ROI_HEIGHT
            # 위 두 직선의 교점의 좌표값 (x2, ROI_HEIGHT)을 구함.               
            x2 = int((ROI_HEIGHT - b_right) / m_right)

            # 두 교점, (x1,0)과 (x2, ROI_HEIGHT)를 잇는 선을 그림
            cv2.line(line_draw_img, (x1,0), (x2,ROI_HEIGHT), Blue, 2)

    #=========================================
    # 차선의 위치를 찾기 위한 기준선(수평선)은 아래와 같음.
    #   (직선의 방정식) y = L_ROW 
    # 위에서 구한 2개의 대표직선, 
    #   (직선의 방정식) y = (m_left)x + (b_left)
    #   (직선의 방정식) y = (m_right)x + (b_right)
    # 기준선(수평선)과 대표직선과의 교점인 x_left와 x_right를 찾음.
    #=========================================

    #=========================================        
    # 대표직선의 기울기 값이 0.0이라는 것은 직선을 찾지 못했다는 의미임
    # 이 경우에는 교점 좌표값을 기존 저장해 놨던 값으로 세팅함 
    #=========================================
    if m_left == 0.0:
        x_left = prev_x_left  # 변수에 저장해 놓았던 이전 값을 가져옴

    #=========================================
    # 아래 2개 직선의 교점을 구함
    # (직선의 방정식) y = L_ROW  
    # (직선의 방정식) y = (m_left)x + (b_left)
    #=========================================
    else:
        x_left = int((L_ROW - b_left) / m_left)
                        
    #=========================================
    # 대표직선의 기울기 값이 0.0이라는 것은 직선을 찾지 못했다는 의미임
    # 이 경우에는 교점 좌표값을 기존 저장해 놨던 값으로 세팅함 
    #=========================================
    if m_right == 0.0:
        x_right = prev_x_right  # 변수에 저장해 놓았던 이전 값을 가져옴	
	
    #=========================================
    # 아래 2개 직선의 교점을 구함
    # (직선의 방정식) y = L_ROW  
    # (직선의 방정식) y = (m_right)x + (b_right)
    #=========================================
    else:
        x_right = int((L_ROW - b_right) / m_right)
       
    #=========================================
    # 대표직선의 기울기 값이 0.0이라는 것은 직선을 찾지 못했다는 의미임
    # 이 경우에 반대쪽 차선의 위치 정보를 이용해서 내 위치값을 정함 
    #=========================================

    ############### 차선 간 차이가 380 px 인지 확인 필요 ##############################

    if m_left == 0.0 and m_right != 0.0:
        x_left = x_right - 380

    if m_left != 0.0 and m_right == 0.0:
        x_right = x_left + 380

    # 이번에 구한 값으로 예전 값을 업데이트 함			
    prev_x_left = x_left
    prev_x_right = x_right
	
    # 왼쪽 차선의 위치와 오른쪽 차선의 위치의 중간 위치를 구함
    x_midpoint = (x_left + x_right) // 2 

    #=========================================
    # 디버깅용 이미지 그리기
    # (1) 수평선 그리기 (직선의 방정식) y = L_ROW 
    # (2) 수평선과 왼쪽 대표직선과의 교점 위치에 작은 녹색 사각형 그리기 
    # (3) 수평선과 오른쪽 대표직선과의 교점 위치에 작은 녹색 사각형 그리기 
    # (4) 왼쪽 교점과 오른쪽 교점의 중점 위치에 작은 파란색 사각형 그리기
    # (5) 화면의 중앙점 위치에 작은 빨간색 사각형 그리기 
    #=========================================
    cv2.line(line_draw_img, (0,L_ROW), (WIDTH,L_ROW), Yellow, 2)
    cv2.rectangle(line_draw_img, (x_left-5,L_ROW-5), (x_left+5,L_ROW+5), Green, 4)
    cv2.rectangle(line_draw_img, (x_right-5,L_ROW-5), (x_right+5,L_ROW+5), Green, 4)
    cv2.rectangle(line_draw_img, (x_midpoint-5,L_ROW-5), (x_midpoint+5,L_ROW+5), Blue, 4)
    cv2.rectangle(line_draw_img, (View_Center-5,L_ROW-5), (View_Center+5,L_ROW+5), Red, 4)

    # 위 이미지를 디버깅용 display_img에 overwrite해서 화면에 디스플레이 함
    display_img[ROI_START_ROW:ROI_END_ROW, 0:WIDTH] = line_draw_img

    pid_center = PID(Kp=0.1, Ki=0.02, Kd=0.01, dt=0.03, max_u =50)

    x_center = (x_left + x_right) /2

    angle_center = -int(pid_center.do(640, x_center, 0))
    
    text = 'Xleft=%d, Xright=%d, xcenter=%d'%(x_left, x_right, angle_center)
    # cv2.putText(display_img, text, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)
    # cv2.imshow("Lanes positions", display_img)
    # cv2.waitKey(1)

    return angle_center

motor = None
motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1) # 모터 토픽을 발행할 것임을 선언

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
#=============================================
def start():
    #=========================================
    # 노드를 생성하고, 구독/발행할 토픽들을 선언합니다.
    #=========================================
    rospy.init_node('xycar_self')
    rospy.Subscriber("/usb_cam/image_raw",Image,usbcam_callback, queue_size=1)

    #=========================================
    # 첫번째 토픽이 도착할 때까지 기다립니다.0
    #=========================================
    rospy.wait_for_message("/usb_cam/image_raw", Image)
    print("Camera Ready --------------")
    
    #=========================================
    # 메인 루프 
    #=========================================
    while not rospy.is_shutdown():


        angle_center = lane_detect()
        speed = 7

        drive(angle_center, speed)


        


#=============================================
# 메인함수 호출
# start() 함수가 실질적인 메인함수임. 
#=============================================
if __name__ == '__main__':
    start()
