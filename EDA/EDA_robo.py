import json

import matplotlib.pyplot as plt
import pandas as pd

# Load the JSON file
file_path = (
    "./data/floating waste dataset FINAL COLOR.v2i.coco/train/_annotations.coco.json"
)
with open(file_path, "r") as f:
    data = json.load(f)

# Extract relevant sections
images = pd.DataFrame(data["images"])
categories = pd.DataFrame(data["categories"])
annotations = pd.DataFrame(data["annotations"])

# Analyze categories - frequency of objects per category
category_freq = annotations["category_id"].value_counts().reset_index()
category_freq.columns = ["category_id", "count"]

# Merge category info
category_freq = category_freq.merge(
    categories[["id", "name"]], left_on="category_id", right_on="id"
)

# Calculate bounding box sizes (width * height)
annotations["bbox_size"] = annotations["bbox"].apply(lambda x: x[2] * x[3])

# Create bins for object sizes (small, medium, large)
bins = [0, 1000, 5000, 10000, 50000, 100000]  # Adjusted for object size distribution
labels = ["Very Small", "Small", "Medium", "Large", "Very Large"]
annotations["size_bin"] = pd.cut(annotations["bbox_size"], bins=bins, labels=labels)

# Frequency of objects in each size bin
size_bin_freq = annotations["size_bin"].value_counts().reset_index()
size_bin_freq.columns = ["size_bin", "count"]

# Plot for category frequency
plt.figure(figsize=(10, 6))
plt.barh(category_freq["name"], category_freq["count"], color="skyblue")
plt.xlabel("Number of Annotations")
plt.ylabel("Categories")
plt.title("Frequency of Object Categories")
plt.gca().invert_yaxis()  # To display the largest category on top
plt.show()

# Plot for bounding box size distribution
plt.figure(figsize=(10, 6))
plt.bar(size_bin_freq["size_bin"], size_bin_freq["count"], color="lightcoral")
plt.xlabel("Object Size Bin")
plt.ylabel("Number of Annotations")
plt.title("Distribution of Object Sizes by Bounding Box")
plt.show()
