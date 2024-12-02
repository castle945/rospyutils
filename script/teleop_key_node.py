#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import curses

def main(stdscr):
    str_pub = rospy.Publisher("/teleop_key", String, queue_size=10)

    while True:
        keycode = stdscr.getch() # 阻塞，等待键盘输入
        
        keymap = {curses.KEY_UP:'w', curses.KEY_DOWN:'s', curses.KEY_LEFT:'a', curses.KEY_RIGHT:'d'}
        keycode = ord(keymap[keycode]) if keycode in keymap.keys() else keycode

        if keycode == ord('q'):
            break
        elif ord('A') <= keycode <= ord('z') or keycode == ord(' '):
            print(chr(keycode))
            str_pub.publish(chr(keycode))

if __name__ == '__main__':
    rospy.init_node('teleop_key_node', anonymous=True)

    curses.wrapper(main)