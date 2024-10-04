from odometry import go_to_xya



while True:
    input = input("Enter x y theta: ")
    x, y, theta = input.split()
    go_to_xya(float(x), float(y), float(theta))
    print("Done")
