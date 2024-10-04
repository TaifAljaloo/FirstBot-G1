FILE_NAME = "logs.txt"

class Logs:
    file = None

    def __init__(self):
        self.file = open(FILE_NAME, "w")
    
    def set_color(self, color):
        self.file.write(f"# {color}\n")

    def set_data(self, left, right, time):
        self.file.write(f"{left} {right} {time}\n")
    
    def gen_plt(self):
        self.file.close()
        with open(FILE_NAME, "r") as f: data = [_.replace("\n", "").split(" ") for _ in f.readlines()]

        l_data, l_pos = data[1], [0,0,0]
        c_color = data[0][1]
        col = {c_color: []}

        for i in range(2, len(data)):
            if data[i][0] == "#":
                c_color = data[i][1]
                col[c_color] = []
            else:
                vl, va = direct_kinematics(float(data[i][0]), float(data[i][1]))
                dx, dy, dtheta = tick_odom(l_pos[0], l_pos[1], l_pos[2], vl, va, float(data[i][2]) - float(l_data[2]))
                col[c_color].append([dx, dy, dtheta])
                l_data, l_pos = data[i], [dx, dy, dtheta]

        for c in col.keys():
            plt.scatter([p[0] for p in col[c]], [p[1] for p in col[c]], color=c)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.title('Trajet du robot')
        plt.show()

    def __del__(self):
        if self.file != None:
            self.file.close()