import cv2
import time
import numpy as np
import motors
from enum import Enum
import signal
import sys
import logs


# Constants
lower_red = (0, 90, 90)
upper_red = (190, 255, 255)
lower_black = (0, 0, 0)
upper_black = (180, 255, 120)

# Global variables
last_time = 0
current_state = 0
last_turn = 0
testing = False


def is_yellow_present(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    return np.any(mask)

def process_frame(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv, 5)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray

def display_histogram(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [1, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    hist_img = np.zeros((300, 256), dtype=np.uint8)
    for x in range(1, 256):
        cv2.line(hist_img, (x-1, 300 - int(hist[x-1])), (x, 300 - int(hist[x])), (255,), 1)
    cv2.imshow('Histogram', hist_img)

def calculate_sums(gray):
    parts = [gray[:, i*gray.shape[1]//6:(i+1)*gray.shape[1]//6] for i in range(6)]
    sums = [np.sum(part)/100000 * 4 for part in parts]
    return sums

def control_motor(sums, motor):
    global last_turn
    left = sum(sums[:3]) / 10
    right = sum(sums[3:]) / 10
    total_sum = sum(sums)
    if left > right and left > 3:
        last_turn = 0
    elif right > left and right > 3:
        last_turn = 1
    if left + right < 3:
        if last_turn:
            right = 3
        else:
            left = 3
    if not testing:
        motor.move(right, left)
    return left, right

def main():
    global last_time, current_state

    motor = motors.motor()

    # Try different camera indices if the default one doesn't work
    camera_indices = [0, 1, 2]
    cam = None
    for index in camera_indices:
        cam = cv2.VideoCapture(index)
        if cam.isOpened():
            print(f"Camera opened successfully with index {index}")
            break
        else:
            cam.release()
            cam = None

    if cam is None:
        print("Error: Could not open any camera.")
        exit()

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    logs = Logs()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        if is_yellow_present(frame):
            if time.time() - last_time > 20:
                current_state += 1
                print("Yellow detected= ", current_state)
                last_time = time.time()

        if current_state == 1:
            print("Black detected")
            logs.set_color("black")
            frame, gray = process_frame(frame, lower_black, upper_black)
            frame[np.where((frame == [0, 0, 0]).all(axis=2))] = [255, 0, 0]
            cv2.imshow('Camera', gray)
            display_histogram(gray)
            sums = calculate_sums(gray)
            left, right = control_motor(sums, motor)
            print("Left Black: ", left, " Right Black: ", right)
            logs.set_data(left, right, time.time())

        if current_state == 2:
            print("Red detected")
            logs.set_color("red")
            frame, gray = process_frame(frame, lower_red, upper_red)
            cv2.imshow('Camera', gray)
            display_histogram(gray)
            sums = calculate_sums(gray)
            left, right = control_motor(sums, motor)
            print("Left Red: ", left, " Right Red: ", right)
            logs.set_data(left, right, time.time())

        if current_state == 3:
            motor.stop()
            motor.unclock()
            print("Finished")
            cam.release()
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    # logs.gen_plt()

if __name__ == "__main__":
    main()