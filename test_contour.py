import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import motors


last_turn = 0
isBlack = 0
testing = True
# Open the default camera
cam = cv2.VideoCapture(0)

if(not testing):
  motor = motors.motor()
# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')



fps = 0
num_frames = 30

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
  
  lower_black = (0, 0, 0)
  upper_black = (360, 100, 10)
  
  lower_red = (0, 90, 90)
  upper_red = (190, 255, 255)
  if(not isBlack):
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # No need for a second mask for blue as it doesn't wrap around the HSV spectrum
    mask2 = mask1

    # Combine the masks
    mask = mask1 | mask2

    # Bitwise-AND mask and original image
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Write the frame to the output file
    # out.write(gray)


    # Display the captured frame and histogram
    cv2.imshow('Camera', gray)
    threshold1 = 100
    threshold2 = 200

    edges = cv2.Canny(gray, threshold1, threshold2)

    contours, hierarchies = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    longest_contour = None
    longest_contour_length = 0

    for contour in contours:
        contour_length = cv2.arcLength(contour, True)
        if contour_length > longest_contour_length:
            longest_contour = contour
            longest_contour_length = contour_length

    if longest_contour is not None:
        cv2.drawContours(frame, [longest_contour], -1, (0, 255, 0), 2)

    cv2.imshow('Edge Detection and Longest Contour', frame)

  


  # Press 'q' to exit the loop
  if cv2.waitKey(1) == ord('q'):
    break


# Release the capture and writer objects
cam.release()
# out.release()
cv2.destroyAllWindows()
