import motors
import time
from math import pi

# user writes a command in the terminal and the robot moves accordingly

def main():
    needed_time = 100/(5.1*pi)
    motor = motors.motor()
    motor.move_forward(2*pi)
    time.sleep(needed_time)
    motor.stop()
        
if __name__ == "__main__":
    main()