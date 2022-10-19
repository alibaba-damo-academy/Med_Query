# Copyright (c) DAMO Health


def convert_df_to_dicts(case_df):
    case_dicts = []
    for idx, row in case_df.iterrows():
        bbox = {
            "filename": row["filename"],
            "label": row["label"],
            "center_w": [row["world_x"], row["world_y"], row["world_z"]],
            "x_axis_local": [row["dx1"], row["dx2"], row["dx3"]],
            "y_axis_local": [row["dy1"], row["dy2"], row["dy3"]],
            "z_axis_local": [row["dz1"], row["dz2"], row["dz3"]],
        }
        for key in ["width", "height", "depth"]:
            bbox.update({key: row[key]})
        case_dicts.append(bbox)
    return case_dicts
