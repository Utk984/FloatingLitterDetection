import os
import xml.etree.ElementTree as ET

# Define the class-to-index mapping
class_mapping = {"plastic":0, "bottle": 0, "carton": 0, "paper": 0}  # Update this mapping as per your classes

# Input and output directories
input_dir = "./dataset_test/Annotations"
output_dir = "./labels2"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def convert_to_yolo(size, box):
    """
    Convert bounding box to YOLO format
    size: (width, height) of the image
    box: [xmin, ymin, xmax, ymax]
    Returns normalized [x_center, y_center, width, height]
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[2]) / 2.0 * dw
    y_center = (box[1] + box[3]) / 2.0 * dh
    width = (box[2] - box[0]) * dw
    height = (box[3] - box[1]) * dh
    return x_center, y_center, width, height

def convert_xml_to_yolo(xml_file, output_file):
    """
    Convert a single XML annotation file to YOLO format
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image size
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)
    
    # Prepare YOLO format annotations
    yolo_annotations = []
    
    for obj in root.iter("object"):
        class_name = obj.find("name").text
        if class_name not in class_mapping:
            continue  # Skip if class is not in the mapping
        
        class_id = class_mapping[class_name]
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        # Convert to YOLO format
        x_center, y_center, width, height = convert_to_yolo(
            (img_width, img_height), (xmin, ymin, xmax, ymax)
        )
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    # Write to output file
    with open(output_file, "w") as f:
        f.write("\n".join(yolo_annotations))

# Process all XML files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".xml"):
        xml_path = os.path.join(input_dir, file_name)
        txt_file_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_file_name)
        convert_xml_to_yolo(xml_path, txt_path)

print("Conversion completed. YOLO annotations are saved in:", output_dir)

