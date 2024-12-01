import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import Normalize, ToTensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class ATLANTIS(data.Dataset):
    def __init__(self, root, split, output_dir=None):
        super(ATLANTIS, self).__init__()
        self.root = root
        self.split = split
        self.output_dir = output_dir
        self.images_base = os.path.join(self.root, "images", self.split)
        self.masks_base = os.path.join(self.root, "masks", self.split)
        self.items_list = self.get_images_list(self.images_base, self.masks_base)

        self.image_transforms = ToTensor()
        self.label_transforms = MaskToTensor()

        self.water_classes = {7, 16, 20, 34, 35, 38, 56, 55}  # Water classes
        self.water_class = 1
        self.ignore_label = 255

        # Create output directory if saving masks
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_images_list(self, images_base, masks_base):
        items_list = []
        for root, dirs, files in os.walk(images_base, topdown=True):
            mask_root = os.path.join(masks_base, os.path.split(root)[1])
            for name in files:
                if name.endswith(".jpg"):
                    mask_name = name.replace(".jpg", ".png")
                    img_file = os.path.join(root, name)
                    lbl_file = os.path.join(mask_root, mask_name)
                    items_list.append(
                        {"image": img_file, "label": lbl_file, "name": name}
                    )
        return items_list

    def __getitem__(self, index):
        image_path = self.items_list[index]["image"]
        label_path = self.items_list[index]["label"]
        name = self.items_list[index]["name"]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        image = self.image_transforms(image)
        label = self.label_transforms(label)

        # Remap water-related classes to 1 and all others to ignore_label (255)
        label_copy = self.ignore_label * torch.ones_like(
            label
        )  # Initialize with ignore_label
        for water_class in self.water_classes:
            label_copy[label == water_class] = self.water_class

        # Save the processed image and mask in the output directory
        if self.output_dir:
            image_save_path = os.path.join(self.output_dir, self.split, name)
            mask_save_path = os.path.join(
                self.output_dir, self.split, name.replace(".jpg", ".png")
            )

            # Save the processed image
            image_pil = Image.fromarray(
                image.permute(1, 2, 0).byte().numpy()
            )  # Ensure correct data type
            image_pil.save(image_save_path)

            # Save the remapped mask
            mask_to_save = label_copy.numpy().astype(np.uint8)
            Image.fromarray(mask_to_save).save(mask_save_path)

        return image, label_copy, name

    def __len__(self):
        return len(self.items_list)


def main():
    dataset_root = "./atlantis"
    output_mask_dir = (
        "./atlantis/processed_data"  # New directory for saving images and masks
    )

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    # Process for all splits (train, val, test)
    for split in ["train", "val", "test"]:
        print(f"Processing {split} split...")
        dataset = ATLANTIS(dataset_root, split, output_dir=output_mask_dir)
        print(f"Number of images in {split}: {len(dataset)}")

        # Create sub-directory for each split (train, val, test)
        split_output_dir = os.path.join(output_mask_dir, split)
        if not os.path.exists(split_output_dir):
            os.makedirs(split_output_dir)

        # Iterate through dataset to save all transformed images and masks
        for i in range(len(dataset)):
            image, mask, name = dataset[i]
            print(f"Processed and saved image and mask for {name} in {split} split.")


if __name__ == "__main__":
    main()
