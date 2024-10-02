import cv2
import time
import numpy as np
import argparse
import motors
from math import pi

# Open the default camera
cam = cv2.VideoCapture(0)
# cam = cv2.VideoCapture()
# cam.open("/dev/v4l/by-id/usb-046d_0825_C9049F60-video-index0")


# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = 0
num_frames = 306
last_turn = 0 #1 -> right, 2 -> left
minimum_speed = 3
speed = 6
motor = motors.motor()

scale_size = 100

right_turn_f = 0
left_turn_f = 0

while True:
  if fps == 0:
    start = time.time()

  ret, frame = cam.read()

  fps += 1

  if fps == num_frames:
    end = time.time()
    seconds = end - start
    print("FPS: ", num_frames / seconds)
    fps = 0

  width = 640
  height = 480
  # resize = cv2.resize(frame, (width, height))

  # Convert the frame to HSV color space
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Define the range of red color in HSV
  lower_blue = (100, 150, 0)
  upper_blue = (140, 255, 255)
  mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

  # No need for a second mask for blue as it doesn't wrap around the HSV spectrum
  mask2 = mask1

  ############################################## Detect red color
  # lower_red = (0, 120, 70)
  # upper_red = (10, 255, 255)
  # mask1 = cv2.inRange(hsv, lower_red, upper_red)

  # lower_red = (170, 120, 70)
  # upper_red = (180, 255, 255)
  # mask2 = cv2.inRange(hsv, lower_red, upper_red)
  ##############################################

  # Combine the masks
  mask = mask1 | mask2

  # Bitwise-AND mask and original image
  frame = cv2.bitwise_and(frame, frame, mask=mask)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

  # Write the frame to the output file
  # out.write(gray)


  # Display the captured frame and histogram
  cv2.imshow('Camera', gray)

  # Ignore black color in the histogram
  # Calculate histogram using OpenCV
  hist = cv2.calcHist([gray], [0], None, [256], [1, 256])

  # Normalize the histogram
  cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

  # Create an image to display the histogram
  hist_img = np.zeros((300, 256), dtype=np.uint8)

  # Draw the histogram
  for x in range(1, 256):
    cv2.line(hist_img, (x-1, 300 - int(hist[x-1])), (x, 300 - int(hist[x])), (255,), 1)

  # Display the histogram
  cv2.imshow('Histogram', hist_img)

  print( np.argmax(gray))
  # Calculate the sum of gray values on the left and right halves
  left_half = gray[:, :gray.shape[1] // 2]
  right_half = gray[:, gray.shape[1] // 2:]
  
  
  column_sums = np.sum(gray, axis=0)

  # Find the index of the column with the highest sum
  highest_sum_index = np.argmax(column_sums)

  print("Column index with highest sum:", highest_sum_index)
    
  middle = np.interp(highest_sum_index,[0,320],[-scale_size,scale_size])
  
  print("middle:", middle)
  
  left_sum = np.sum(left_half)
  right_sum = np.sum(right_half)
  total_sum = left_sum + right_sum
  
  if(total_sum == 0):
        left_percentage = 0
        right_percentage = 0
        if(last_turn == 1):
            motor.move(minimum_speed,0)
        else:
            motor.move(0,minimum_speed)
        print("last turn : " + str(last_turn))
          
  else:
    if(middle > 0):
            right_turn_f = middle/scale_size*speed
            left_turn_f = scale_size-middle/scale_size*speed
    else:
            right_turn_f = scale_size-abs(middle)/scale_size*speed
            left_turn_f = abs(middle)/scale_size*speed
          
    print("Left: ", left_turn_f, "% Right: ", right_turn_f, "%")
    motor.move(right_turn_f/100,left_turn_f/100)
  


  # Press 'q' to exit the loop
  if cv2.waitKey(1) == ord('q'):
    break

motor.stop()  
# Release the capture and writer objects
cam.release()
# out.release()
cv2.destroyAllWindows()
