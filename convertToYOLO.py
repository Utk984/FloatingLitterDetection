import os
import pandas as pd

# Paths to dataset directories
base_dir = '/mnt/d/Ed/Plaksha/5th Semester/data/archive'  # Replace with the path to the 'archive' directory
subdirs = ["train", "valid", "test"]

# Function to convert bounding boxes to YOLO format
def convert_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

# Process each subdirectory
for subdir in subdirs:
    annotations_path = os.path.join(base_dir, subdir, "_annotations.csv")
    images_dir = os.path.join(base_dir, subdir)
    labels_dir = os.path.join(base_dir, subdir, "labels")
    
    # Create labels directory if it doesn't exist
    os.makedirs(labels_dir, exist_ok=True)

    # Load annotations CSV
    annotations = pd.read_csv(annotations_path)

    # Group by filename
    grouped = annotations.groupby("filename")

    for filename, group in grouped:
        # Get image dimensions
        img_width = group["width"].iloc[0]
        img_height = group["height"].iloc[0]

        # Create a YOLO format annotation file
        yolo_annotations = []
        for _, row in group.iterrows():
            class_id = 0  # Assuming a single class ('float-trash')
            xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            yolo_bbox = convert_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height)
            yolo_annotations.append(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

        # Save the YOLO annotations to a .txt file
        label_file_path = os.path.join(labels_dir, f"{os.path.splitext(filename)[0]}.txt")
        with open(label_file_path, "w") as f:
            f.writelines(yolo_annotations)

    print(f"Converted annotations for {subdir} directory.")

print("Conversion to YOLO format completed!")
