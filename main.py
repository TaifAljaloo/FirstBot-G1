import motors
import time
from pypot.dynamixel.io.abstract_io import DxlCommunicationError

try:
    motor = motors.motor()
    motor.move(left_value=6, right_value=6) # rad/s
    time.sleep(2)
    motor.stop()
except DxlCommunicationError as e:
    print(f"Communication error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")