import argparse

import matplotlib.pyplot as plt

def load_csv_loss_data(filename):
    """
    Loads the csv data stored at ``fileanme`` and returns an (x,y) tuple
    """
    x, y = [], []
    with open(filename, 'r') as ifile:
        for line in ifile:
            data = line.strip().split(",")
            x.append(data[0])
            y.append(data[1])
    return x, y


def generate_loss_plot(filename, savename):
    """
    Creates a plot of the loss data stored in ``filename``
    """
    t, loss = load_csv_loss_data(filename)
    plt.plot(t, loss, linewidth=0.25)
    plt.savefig(savename, dpi=1000)

def main():
    generate_loss_plot("../build/results_test.csv", "test.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for loss data")
    parser.add_argument("--if", dest="infile", type=str,
            help="Location of loss data file")
    parser.add_argument("--of", dest="outfile", type=str,
            help="Location of the generated graph image")
    args = parser.parse_args()
    generate_loss_plot(args.infile, args.outfile)
