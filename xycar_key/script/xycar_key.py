#!/usr/bin/env python
import rospy
import time
import sys, select, os
from xycar_msgs.msg import xycar_motor

if os.name == 'nt':
  import msvcrt, time
else:
  import tty, termios


motor_control = xycar_motor()

# 키보드값 받기
def getKey():
    if os.name == 'nt':
        timeout = 0.1
        startTime = time.time()
        while(1):
            if msvcrt.kbhit():
                if sys.version_info[0] >= 3:
                    return msvcrt.getch().decode()
                else:
                    return msvcrt.getch()
            elif time.time() - startTime > timeout:
                return ''

    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

# xycar에 angle, speed 퍼블리쉬
def motor_pub(angle, speed): 
    global pub
    global motor_control

    motor_control.angle = angle
    motor_control.speed = speed

    pub.publish(motor_control)

def constrain(input, low, high):
    if input < low:
      input = low
    elif input > high:
      input = high
    else:
      input = input

    return input

if __name__ == '__main__':
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('xycar_key')
    pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    
    speed = 0
    angle = 0

    try:
        print('start')
        rate = rospy.Rate(200) # 10hz
        while not rospy.is_shutdown():
            key = getKey()
            if key == 'w' :
                speed += 1
                speed = constrain(speed,-20,20)
                print(f'speed {speed}, angle: {angle}')
            elif key == 'x' :
                speed += -1
                speed = constrain(speed,-20,20)
                print(f'speed {speed}, angle: {angle}')

            elif key == 'a' :
                angle += -5
                angle = constrain(angle, -45, 45)
                print(f'speed {speed}, angle: {angle}')

            elif key == 'd' :
                angle += 5
                angle = constrain(angle, -45, 45)
                print(f'speed {speed}, angle: {angle}')

            elif key == ' ' or key == 's' :
                speed = 0
                angle = 0
                print(f'speed {speed}, angle: {angle}')
            else:
                if (key == '\x03'):
                    break
            motor_pub(angle, speed)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass

    finally:
        motor_pub(angle=0, speed=0)

    if os.name != 'nt':
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
