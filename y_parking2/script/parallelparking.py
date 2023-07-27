#!/usr/bin/env python
import rospy
from xycar_msgs.msg import xycar_motor

motor_control = xycar_motor()
motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

# motor control 
def motor_controlf(angle, speed):

    global motor_pub
    global motor_control
	 		
    # 모터 메시지 생성
    motor_control.angle = angle
    motor_control.speed = speed

    # 모터 메시지 발행
    motor_pub.publish(motor_control)



def ParallelParking():

    rospy.init_node('y_parking2')
        # ROS 노드 초기화
    rate = rospy.Rate(1)
        
    
    while not rospy.is_shutdown():


        # 50의 속도로 5초 동안 모터를 전진
        # speed = 7
        motor_controlf(0,7)
        print("forward")
        rate.sleep()
        rate.sleep()
        rate.sleep()
        rate.sleep()
        rate.sleep()

        # 5초
        motor_controlf(0,0)
        rate.sleep()
        rate.sleep()
        rate.sleep()
        rate.sleep()
        rate.sleep()
        #rate.sleep(5)

        # 모터 정지
        #speed2 = -7
        #motor_controlf(20, int(speed))
        #print("backward")
        #rate.sleep()

if __name__ == '__main__':
    try:
        ParallelParking()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

