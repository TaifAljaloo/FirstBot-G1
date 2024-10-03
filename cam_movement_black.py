import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import motors

last_turn = 0
testing = True
# Open the default camera
cam = cv2.VideoCapture(0)

if(not testing):
  motor = motors.motor()
# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
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
    
  # Convert the frame to HSV color space
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Reduce light glare by applying a median blur
  hsv = cv2.medianBlur(hsv, 5)

  # Define the range of black color in HSV
  lower_black = (0, 0, 0)
  upper_black = (180, 255, 110)
  
  # Define the range of red color in HSV
  lower_red = (0, 90, 90)
  upper_red = (190, 255, 255)
  
  # Create masks for the colors
  mask_black = cv2.inRange(hsv, lower_black, upper_black)
  mask_red = cv2.inRange(hsv, lower_red, upper_red)

  # Combine the masks
  mask = mask_black

  # Bitwise-AND mask and original image
  frame = cv2.bitwise_and(frame, frame, mask=mask)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

  # Write the frame to the output file
  # out.write(gray)

  # Turn the black color detected into blue
  frame[np.where((frame == [0, 0, 0]).all(axis=2))] = [255, 0, 0]

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

  
  part1 = gray[:, :gray.shape[1] // 6]
  part2 = gray[:, gray.shape[1] // 6: 2 * gray.shape[1] // 6]
  part3 = gray[:, 2 * gray.shape[1] // 6: 3 * gray.shape[1] // 6]
  part4 = gray[:, 3 * gray.shape[1] // 6: 4 * gray.shape[1] // 6]
  part5 = gray[:, 4 * gray.shape[1] // 6: 5 * gray.shape[1] // 6]
  part6 = gray[:, 5 * gray.shape[1] // 6:]
  sum1 = np.sum(part1)/100000 * 6
  sum2 = np.sum(part2)/100000 * 4
  sum3 = np.sum(part3)/100000 * 5
  sum4 = np.sum(part4)/100000 * 5
  sum5 = np.sum(part5)/100000 * 4
  sum6 = np.sum(part6)/100000 * 6

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



  # Press 'q' to exit the loop
  if cv2.waitKey(1) == ord('q'):
    break


# Release the capture and writer objects
cam.release()
# out.release()
cv2.destroyAllWindows()