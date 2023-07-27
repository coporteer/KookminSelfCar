#!/usr/bin/env python
import rospy
import time
from sensor_msgs.msg import Joy
from xycar_msgs.msg import xycar_motor


xbox_axes = [0 for _ in range(8)]
motor_control = xycar_motor()
pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

def cb(data):
    global xbox_axes
    rospy.loginfo(f'{data.axes[0]:5.2f}, {data.axes[1]:5.2f}, {data.axes[4]:5.2f}, {data.axes[3]:5.2f}')
    xbox_axes = data.axes

def motor_pub(angle, speed): 
    global pub
    global motor_control

    motor_control.angle = angle
    motor_control.speed = speed

    pub.publish(motor_control)

def talker():
    rospy.init_node('xycar_joy')
    rospy.Subscriber("joy", Joy, cb)

    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        
        speed = xbox_axes[1]*5
        angle = xbox_axes[3]*5
        motor_pub(angle, speed)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass