import csv
import pandas as pd

# list to store file lines
lines = []
# read file
with open(r"./data/fasttext/fasttext.4B.id.300d.txt", 'r', encoding="utf8") as fp:
    # read an store all lines into list
    lines = fp.readlines()

# Write file
with open(r"./data/fasttext/fasttext.4B.id.300d.txt", 'w', encoding="utf8") as fp:
    # iterate each line
    for number, line in enumerate(lines):
        # delete line 5 and 8. or pass any Nth line you want to remove
        # note list index starts from 0
        if number != 0:
            fp.write(line)
        else:
            print(line)

source_embedding = pd.read_table("./data/fasttext/fasttext.4B.id.300d.txt",
                                     index_col=0,
                                     sep=' ',
                                     header=None,
                                     quoting=csv.QUOTE_NONE,
                                     names=range(301)).drop(300, axis=1)

print(source_embedding.head())