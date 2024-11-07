import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
file_path_csv = "./data/archive/train/_annotations.csv"
df = pd.read_csv(file_path_csv)

# Analyze object class distribution
class_distribution = df["class"].value_counts().reset_index()
class_distribution.columns = ["class", "count"]

# Calculate bounding box width and height
df["bbox_width"] = df["xmax"] - df["xmin"]
df["bbox_height"] = df["ymax"] - df["ymin"]
df["bbox_size"] = df["bbox_width"] * df["bbox_height"]

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
plt.gca().invert_yaxis()  # To display largest class on top
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
