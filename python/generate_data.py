import random

def normalize_data(x):
    norm = sum((i*i for i in x))
    return [i/norm for i in x]

def generate_linear_sep(filename, z=1, dim=1000, n=6, spread=10000):
    with open(filename, "w") as ofile:
        for i in range(n):
            line = ""
            label = random.choice([-1, 1])
            line += str(label) + ","
            vec = [random.randint(-spread//2, spread//2) for _ in range(dim - 1)]
            z_val = 2 if label == 1 else 0
            nvec = normalize_data(vec + [z_val])
            line += ",".join([str(x) for x in nvec])
            ofile.write(line + "\n")

if __name__ == "__main__":
    generate_linear_sep("dim50k_n1000_s100_test.csv", dim=50000, n=1000, spread=100)
