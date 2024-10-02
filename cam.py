import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Open the default camera
cam = cv2.VideoCapture()
cam.open("/dev/v4l/by-id/usb-046d_0825_C9049F60-video-index0")

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))
fps = 0

num_frames = 30
# right_red = 0
# left_red = 0

while True:
  left_red = 0
  right_red = 0

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
  lower_red = (0, 120, 70)
  upper_red = (10, 255, 255)
  mask1 = cv2.inRange(hsv, lower_red, upper_red)

  lower_red = (170, 120, 70)
  upper_red = (180, 255, 255)
  mask2 = cv2.inRange(hsv, lower_red, upper_red)

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
  gray_nonzero = gray[gray > 0]
  plt.hist(gray_nonzero.ravel(), 256, [1, 256], color='gray')
  plt.draw()
  plt.pause(0.1)
  plt.clf()



  # Press 'q' to exit the loop
  if cv2.waitKey(1) == ord('q'):
    break


# Release the capture and writer objects
cam.release()
# out.release()
cv2.destroyAllWindows()
