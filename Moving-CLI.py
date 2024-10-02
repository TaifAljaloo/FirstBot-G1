import motors
import time

# user writes a command in the terminal and the robot moves accordingly

def main():
    motor = motors.motor()
    while True:
        command = input("Enter a command: ")
        if command == "z":
            motor.move_forward(6)
        elif command == "s":
            motor.move_backward(6)
        elif command == "q":
            motor.move_left(6)
        elif command == "d":
            motor.move_right(6)
        elif command == "a":
            motor.move_forward_left(6)
        elif command == "e":
            motor.move_forward_right(6)
        elif command == "w":
            motor.move_backward_left(6)
        elif command == "c":
            motor.move_backward_right(6)
        elif command == "x":
            motor.stop()
        else:
            print("Invalid command")
            continue
        time.sleep(2)
        motor.stop()