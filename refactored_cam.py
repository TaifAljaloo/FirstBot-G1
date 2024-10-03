import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import motors


class Filter:
    def __init__(self,name,lower_limit,higher_limit):
        self.name = name
        self.lower_limit = lower_limit
        self.higher_limit = higher_limit
    def getLL(self):
        return self.lower_limit
    def getHL(self):
        return self.higher_limit
    def getName(self):
        return self.name
      
      
class LineFollower:
      def __init__(self,minimum_speed,max_speed,robot):
            self.max_speed = max_speed
            self.min_speed = minimum_speed
            self.robot = robot
      def setSpeed(self,left,right):
            self.robot.move(left,right)

def setup_cam(width,height):
  cam = cv2.VideoCapture(0)
  frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
  
  cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
  cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  return cam 
      

def get_histogramme_from_frame(hist,filter):
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Define the range of red color in HSV
  mask1 = cv2.inRange(hsv, filter.lower_limit, filter.higher_limit)

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

  # Ignore black color in the histogram
  # Calculate histogram using OpenCV
  hist = cv2.calcHist([gray], [0], None, [256], [1, 256]) 

  # Normalize the histogram
  cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
  return hist,gray

def get_speeds():
  # Convert the frame to  HSV color space
  hist,gray = get_histogramme_from_frame()
  # Create an image to display the histogram
  hist_img = np.zeros((300, 256), dtype=np.uint8)

  # Draw the histogram
  for x in range(1, 256):
    cv2.line(hist_img, (x-1, 300 - int(hist[x-1])), (x, 300 - int(hist[x])), (255,), 1)

  # Display the histogram
  cv2.imshow('Histogram', hist_img)

  # Calculate the sum of gray values in six equal parts
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
  left = left/ 100
  left = round(left,2)
  right = sum4 + sum5 + sum6
  right = right/100
  right = round(right,2)
  
  return left,right



def main():
  cam = setup_cam(320,240)      
  black_filter = Filter("black",(0, 0, 0),(360, 100, 10))
  red_filter = Filter("red",(0, 90, 90),(190, 255, 255))
  robo = LineFollower(3,12,motors.motor())
  
  while(True):
        print(get_speeds())
        robo.setSpeed(get_speeds())
        
  cam.release()
  cv2.destroyAllWindows()
        
  
      
      


if __name__ == '__main__':
    main()
    