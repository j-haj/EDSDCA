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

def generate_loss_plots(filename1, data_name1, filename2, data_name2, title,
        savename, xmax=None):
    """
    creates a plot of the loss data stored in ``filename``
    """
    t1, loss1 = load_csv_loss_data(filename1)
    t2, loss2 = load_csv_loss_data(filename2)

    plt.plot(t1, loss1, linewidth=0.25, label=data_name1)
    plt.plot(t2, loss2, linewidth=0.25, label=data_name2)

    if xmax is not None:
        plt.xlim([0, xmax])
    plt.xlabel("runtime (s)")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.savefig(savename, dpi=600)



def generate_loss_plot(filename, savename):
    """
    creates a plot of the loss data stored in ``filename``
    """
    t, loss = load_csv_loss_data(filename)
    plt.plot(t, loss, linewidth=0.25)
    plt.xlabel("runtime (s)")
    plt.ylabel("loss")
    plt.title("cumulative training loss vs runtime")
    plt.savefig(savename, dpi=1000)

def main():
    generate_loss_plot("../build/results_test.csv", "test.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for loss data")
    parser.add_argument("--if", dest="infile", type=str,
            help="Location of loss data file")
    parser.add_argument("--of", dest="outfile", type=str,
            help="Location of the generated graph image")

    parser.add_argument("--if1", dest="infile1", type=str,
            help="Location of loss data for file 1")
    parser.add_argument("--l1", dest="label1", type=str,
            help="Label for data in if1")
    parser.add_argument("--if2", dest="infile2", type=str,
            help="Location of loss data for file 2")
    parser.add_argument("--l2", dest="label2", type=str,
            help="Label for data in if2")
    parser.add_argument("--title", dest="title", type=str,
            help="Title for plot")
    parser.add_argument("--xmax", dest="xmax", type=int,
            help="Maximum value for x axis")

    args = parser.parse_args()
    if args.infile != None:
        generate_loss_plot(args.infile, args.outfile)
    else:
        generate_loss_plots(args.infile1, args.label1, args.infile2,
                args.label2, args.title, args.outfile, args.xmax)


