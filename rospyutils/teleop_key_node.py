#!/usr/bin/env python
import rclpy
from std_msgs.msg import String
import curses

def main_loop(stdscr):
    node = rclpy.create_node('teleop_key_node')
    str_pub = node.create_publisher(String, "/teleop_key", qos_profile=10)

    msg = String()
    while rclpy.ok():
        keycode = stdscr.getch() # 阻塞，等待键盘输入，故不需要 spin
        
        keymap = {curses.KEY_UP:'w', curses.KEY_DOWN:'s', curses.KEY_LEFT:'a', curses.KEY_RIGHT:'d'}
        keycode = ord(keymap[keycode]) if keycode in keymap.keys() else keycode

        if keycode == ord('q'):
            break
        elif ord('A') <= keycode <= ord('z') or keycode == ord(' '):
            print(chr(keycode))
            msg.data = chr(keycode)
            str_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    try:
        curses.wrapper(main_loop)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()