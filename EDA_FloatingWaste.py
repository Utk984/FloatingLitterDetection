import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pandas as pd


# Function to parse an XML file and extract relevant data
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    objects = []
    for obj in root.findall("object"):
        obj_class = obj.find("name").text
        xmin = int(obj.find("bndbox/xmin").text)
        ymin = int(obj.find("bndbox/ymin").text)
        xmax = int(obj.find("bndbox/xmax").text)
        ymax = int(obj.find("bndbox/ymax").text)
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        bbox_size = bbox_width * bbox_height
        objects.append(
            [
                filename,
                width,
                height,
                obj_class,
                xmin,
                ymin,
                xmax,
                ymax,
                bbox_width,
                bbox_height,
                bbox_size,
            ]
        )

    return objects


# Path to the folder containing XML files
xml_folder = "./data/FloatingWaste-I/Annotations/"

# Gather data from all XML files
all_data = []
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(xml_folder, xml_file)
        all_data.extend(parse_xml(xml_path))

# Convert to a pandas DataFrame
columns = [
    "filename",
    "width",
    "height",
    "class",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "bbox_width",
    "bbox_height",
    "bbox_size",
]
df = pd.DataFrame(all_data, columns=columns)

# EDA: Class Distribution
class_distribution = df["class"].value_counts().reset_index()
class_distribution.columns = ["class", "count"]

# Create bins for bounding box sizes
bins = [0, 5000, 20000, 50000, 100000, 200000]  # Adjust for object size distribution
labels = ["Very Small", "Small", "Medium", "Large", "Very Large"]
df["size_bin"] = pd.cut(df["bbox_size"], bins=bins, labels=labels)

# Frequency of objects in each size bin
size_bin_distribution = df["size_bin"].value_counts().reset_index()
size_bin_distribution.columns = ["size_bin", "count"]

# Plot for class distribution
plt.figure(figsize=(10, 6))
plt.barh(class_distribution["class"], class_distribution["count"], color="skyblue")
plt.xlabel("Number of Instances")
plt.ylabel("Object Class")
plt.title("Object Class Distribution")
plt.gca().invert_yaxis()  # Largest class on top
plt.show()

# Plot for bounding box size distribution
plt.figure(figsize=(10, 6))
plt.bar(
    size_bin_distribution["size_bin"],
    size_bin_distribution["count"],
    color="lightcoral",
)
plt.xlabel("Object Size Bin")
plt.ylabel("Number of Annotations")
plt.title("Bounding Box Size Distribution")
plt.show()
