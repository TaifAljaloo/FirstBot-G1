import cv2
import time
import numpy as np
import motors
from enum import Enum

lower_red = (0, 90, 90)
upper_red = (190, 255, 255)
lower_black = (0, 0, 0)
upper_black = (180, 255, 120)
last_time = 0
current_state = 0
last_turn = 0
testing = False
choose_color = 0

def is_yellow_present(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    return np.any(mask)

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

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
start_time = time.time()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

while True:
    ret, frame = cam.read()
    

    if not ret:
        print("Error: Failed to capture image.")
        break

    if is_yellow_present(frame):
        print("Yellow detected")
        if time.time() - last_time > 20:
            current_state += 1
            last_time = time.time()

    if current_state == 1 and choose_color == 0:
        #print("Black detected")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.medianBlur(hsv, 3)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask = mask_black
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame[np.where((frame == [0, 0, 0]).all(axis=2))] = [255, 0, 0]
        cv2.imshow('Camera', gray)
        hist = cv2.calcHist([gray], [0], None, [256], [1, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        hist_img = np.zeros((300, 256), dtype=np.uint8)

        for x in range(1, 256):
            cv2.line(hist_img, (x-1, 300 - int(hist[x-1])), (x, 300 - int(hist[x])), (255,), 1)

        cv2.imshow('Histogram', hist_img)

        part1 = gray[:, :gray.shape[1] // 6]
        part2 = gray[:, gray.shape[1] // 6: 2 * gray.shape[1] // 6]
        part3 = gray[:, 2 * gray.shape[1] // 6: 3 * gray.shape[1] // 6]
        part4 = gray[:, 3 * gray.shape[1] // 6: 4 * gray.shape[1] // 6]
        part5 = gray[:, 4 * gray.shape[1] // 6: 5 * gray.shape[1] // 6]
        part6 = gray[:, 5 * gray.shape[1] // 6:]
        sum1 = np.sum(part1)/100000 * 3
        sum2 = np.sum(part2)/100000 * 3
        sum3 = np.sum(part3)/100000 * 3
        sum4 = np.sum(part4)/100000 * 3
        sum5 = np.sum(part5)/100000 * 3
        sum6 = np.sum(part6)/100000 * 3
        left = sum1 + sum2 + sum3
        left = left/ 10
        left = round(left,2)
        right = sum4 + sum5 + sum6
        right = right/10
        right = round(right,2)
        print()
        total_sum = sum1 + sum2 + sum3 + sum4 + sum5 + sum6
        if(left > right and left > 3):
          last_turn = 0;
        elif(right> left and right > 3):
          last_turn = 1;
        if(left+right < 3 ):
          if(last_turn):
                right = 3
          else:
                left = 3
                print("left :"+ str(left))
        print("right :"+ str(right))
        if(not testing):
              motor.move(right,left)

    if current_state == 2 and choose_color == 0:
        choose_color = 1
        print("Red detected")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        mask2 = mask1
        mask = mask1 | mask2
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Camera', gray)

        hist = cv2.calcHist([gray], [0], None, [256], [1, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        hist_img = np.zeros((300, 256), dtype=np.uint8)

        for x in range(1, 256):
            cv2.line(hist_img, (x-1, 300 - int(hist[x-1])), (x, 300 - int(hist[x])), (255,), 1)

        cv2.imshow('Histogram', hist_img)

        part1 = gray[:, :gray.shape[1] // 6]
        part2 = gray[:, gray.shape[1] // 6: 2 * gray.shape[1] // 6]
        part3 = gray[:, 2 * gray.shape[1] // 6: 3 * gray.shape[1] // 6]
        part4 = gray[:, 3 * gray.shape[1] // 6: 4 * gray.shape[1] // 6]
        part5 = gray[:, 4 * gray.shape[1] // 6: 5 * gray.shape[1] // 6]
        part6 = gray[:, 5 * gray.shape[1] // 6:]
        sum1 = np.sum(part1) / 100000 * 6
        sum2 = np.sum(part2) / 100000 * 4
        sum3 = np.sum(part3) / 100000 * 5
        sum4 = np.sum(part4) / 100000 * 5
        sum5 = np.sum(part5) / 100000 * 4
        sum6 = np.sum(part6) / 100000 * 6
        left = sum1 + sum2 + sum3
        left = left / 100
        left = round(left, 2)
        right = sum4 + sum5 + sum6
        right = right / 100
        right = round(right, 2)

        print()
        total_sum = sum1 + sum2 + sum3 + sum4 + sum5 + sum6
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
            motor.move(left, right)

    if current_state == 3 and choose_color == 1:
        motor.stop()
        motor.unclock()
        print("Finished")
        cam.release()
        cv2.destroyAllWindows()
        break

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break