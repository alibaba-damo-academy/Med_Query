# Copyright (c) DAMO Health

import os


def get_itksnap_color_dict():
    color_table_file = os.path.join(os.path.dirname(__file__), "itksnap_color_table.txt")
    with open(color_table_file, "r") as f:
        lines = f.readlines()

    color_dict = dict()
    for line in lines:
        if line.startswith("#"):
            continue
        words = line.split()
        key = int(words[0])
        color = [int(x) for x in words[1:4]]
        color_dict.update({key: color})
    return color_dict
