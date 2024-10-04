from odometry import go_to_xya



while True:
    raw = input("Enter x y theta: ")
    x, y, theta = raw.split()
    go_to_xya(float(x), float(y), float(theta))
    print("Done")
