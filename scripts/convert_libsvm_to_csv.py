import os

def convert_to_csv(path):
    output_file = os.path.join(path, ".csv")

    with open(path, "r") as infile:
        with open(output_file, "w") as outfile:
            for line in infile:
                split_line = line.split()
                
