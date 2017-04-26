import random

def generate_linear_sep(filename, z=1, dim=3, n=6, spread=10000):
    with open(filename, "w") as ofile:
        for i in range(n):
            line = ""
            label = random.choice([-1, 1])
            line += str(label)
            for i in range(dim - 1):
                x = random.randint(-spread//2, spread//2)
                line += "," + str(x)
            if label == 1:
                line += "," + str(z + 1)
            else:
                line += "," + str(z - 1)
            ofile.write(line + "\n")

if __name__ == "__main__":
    generate_linear_sep("n50_s100_test.csv", n=100, spread=100)
