import os
import cv2

def resize_images_and_bboxes(image_folder, label_folder, output_image_folder, output_label_folder, target_size=640):
    # Create output directories if they don't exist
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):  # Check for valid image file extensions
            image_path = os.path.join(image_folder, image_name)
            label_path = os.path.join(label_folder, os.path.splitext(image_name)[0] + ".txt")
            
            # Check if the corresponding label file exists
            if not os.path.exists(label_path):
                print(f"Label file not found for {image_name}. Skipping...")
                continue

            # Read the image
            image = cv2.imread(image_path)
            h_orig, w_orig = image.shape[:2]

            # Resize the image
            image_resized = cv2.resize(image, (target_size, target_size))
            output_image_path = os.path.join(output_image_folder, image_name)
            cv2.imwrite(output_image_path, image_resized)

            # Read and process the label file
            with open(label_path, 'r') as label_file:
                lines = label_file.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"Invalid label format in {label_path}. Skipping line...")
                    continue

                class_id, x_center, y_center, width, height = map(float, parts)
                # Convert normalized coordinates to absolute
                x_abs = x_center * w_orig
                y_abs = y_center * h_orig
                w_abs = width * w_orig
                h_abs = height * h_orig

                # Scale to new size
                x_abs_scaled = x_abs * (target_size / w_orig)
                y_abs_scaled = y_abs * (target_size / h_orig)
                w_abs_scaled = w_abs * (target_size / w_orig)
                h_abs_scaled = h_abs * (target_size / h_orig)

                # Normalize to new size
                x_center_new = x_abs_scaled / target_size
                y_center_new = y_abs_scaled / target_size
                width_new = w_abs_scaled / target_size
                height_new = h_abs_scaled / target_size

                updated_lines.append(f"{int(class_id)} {x_center_new} {y_center_new} {width_new} {height_new}\n")

            # Write updated labels to output folder
            output_label_path = os.path.join(output_label_folder, os.path.splitext(image_name)[0] + ".txt")
            with open(output_label_path, 'w') as output_label_file:
                output_label_file.writelines(updated_lines)

            print(f"Processed {image_name}")

# Specify paths
image_folder = "datasets/FlowIMG_FloatingWaste/images/train"
label_folder = "datasets/FlowIMG_FloatingWaste/labels/train"
output_image_folder = "datasets/FlowIMG_FloatingWaste/images/train_resized"
output_label_folder = "datasets/FlowIMG_FloatingWaste/labels/train_resized"

resize_images_and_bboxes(image_folder, label_folder, output_image_folder, output_label_folder)

image_folder = "datasets/FlowIMG_FloatingWaste/images/val"
label_folder = "datasets/FlowIMG_FloatingWaste/labels/val"
output_image_folder = "datasets/FlowIMG_FloatingWaste/images/val_resized"
output_label_folder = "datasets/FlowIMG_FloatingWaste/labels/val_resized"

resize_images_and_bboxes(image_folder, label_folder, output_image_folder, output_label_folder)
