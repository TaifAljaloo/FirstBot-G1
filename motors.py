import pypot.dynamixel
import math
import time
    

class motor:

    dxl_io = None

    def __init__(self):
        ports = pypot.dynamixel.get_available_ports()
        print(ports)
        if not ports:
            exit('No port')

        self.dxl_io = pypot.dynamixel.DxlIO(ports[0])
        self.dxl_io.set_wheel_mode([1])

        self.left = 2
        self.right = 1

    def move(self, left_value, right_value):
        self.dxl_io.set_moving_speed({self.left: math.degrees(left_value)})
        self.dxl_io.set_moving_speed({self.right: math.degrees(l-right_value)})

    def stop(self):
        self.dxl_io.set_moving_speed({self.left: math.degrees(0)})
        self.dxl_io.set_moving_speed({self.right: math.degrees(0)})

    def lock(self):
        self.dxl_io.enable_torque([self.left,self.right])

    def unclock(self):
        self.dxl_io.disable_torque([self.left,self.right])

    def get_speed(self):
        speeds = self.dxl_io.get_moving_speed([self.left,self.right])
        left_speed, right_speed = speeds[0], speeds[1]
        return left_speed, right_speed
    
    def move_forward(self, speed):
        self.move(speed, speed)
        
    def move_backward(self, speed):
        self.move(-speed, -speed)
        
    def move_left(self, speed):
        self.move(-speed, speed)
        
    def move_right(self, speed):
        self.move(speed, -speed)
        
    def move_forward_left(self, speed):
        self.move(0, speed)
        
    def move_forward_right(self, speed):
        self.move(speed, 0)
        
    def move_backward_left(self, speed):
        self.move(-speed, 0)
        
    def move_backward_right(self, speed):
        
        self.move(0, -speed)
        
        
        
        
        

    


