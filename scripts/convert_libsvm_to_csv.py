import os


def convert_to_csv(path):
    """
    Converts the file at ``path`` to a csv file
    """
    output_file = path + ".csv"

    with open(path, "r") as infile:
        with open(output_file, "w") as outfile:
            for line in infile:
                split_line = line.strip().split()
                write_line = ""
                for v in split_line:
                    write_line += v
                    write_line += ","
                outfile.write(write_line.strip(",") + "\n")

def convert_libsvm_to_csv(path, num_features):
    """
    Converts the sparse libsvm file to a dense, csv file
    """
    input_file = os.path.join(os.getcwd(), path)
    output_file = os.path.basename(path) + ".csv"

    line_num = 0
    with open(path, "r") as infile:
        with open(output_file, "w") as outfile:
            for line in infile:
                split_line = line.strip().split()
                write_line = ""
                col_num = 0
                for v in split_line:
                    if ":" in v:
                        # handle element
                        first_val, second_val = v.split(":")
                        first_val = int(first_val)
                        if first_val  > col_num:
                            while first_val > col_num:
                                write_line += "0,"
                                col_num += 1
                            col_num += 1
                        else:
                            col_num += 1
                        write_line += second_val + ","
                    else:
                        write_line += v.strip("+") + ","
                        col_num += 1
                if col_num < num_features:
                    while col_num <= num_features:
                        write_line += "0,"
                        col_num += 1
                outfile.write(write_line.strip(",") + "\n")
                line_num += 1

if __name__ == "__main__":
    file_path = "/Users/jhaj/Documents/university-of-iowa/classes/s2017/machine-learning/EDSDCA/data/news20.binary"
    convert_to_csv(file_path)
