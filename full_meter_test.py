import motors
import time
from math import pi

# user writes a command in the terminal and the robot moves accordingly

def main():
    motor = motors.motor()
    motor.move_forward(2*pi)
    time.sleep(6.1)
    motor.stop()
        
if __name__ == "__main__":
    main()