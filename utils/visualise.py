import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def overlay_mask_on_image(image_path, mask_path, output_path=None):
    """
    Overlay the binary mask on the image to visualize.
    The mask will be overlaid with a semi-transparent red color.
    Args:
    - image_path (str): Path to the saved image.
    - mask_path (str): Path to the saved mask.
    - output_path (str): Path to save the overlaid image (optional).
    """
    # Open the image and mask
    image = Image.open(image_path).convert("RGBA")  # Open image as RGBA
    mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale (L mode)

    # Convert mask to a numpy array (0 and 255 values)
    mask_array = np.array(mask)

    # Create a red mask to overlay
    red_mask = np.zeros_like(mask_array)
    red_mask[mask_array == 255] = (
        255  # Only keep the white pixels from the mask (water areas)
    )

    # Create a red-colored version of the mask (RGB format)
    red_mask_image = Image.fromarray(
        np.stack([red_mask, np.zeros_like(red_mask), np.zeros_like(red_mask)], axis=-1)
    )  # RGB red mask

    # Convert red_mask_image to RGBA (adding an alpha channel)
    red_mask_rgba = red_mask_image.convert("RGBA")

    # Overlay the red mask on the image (50% transparency)
    image_array = np.array(image)
    red_mask_rgba_array = np.array(red_mask_rgba)

    # Apply the red mask with transparency (blend RGB channels)
    overlay = red_mask_rgba_array[:, :, 0] > 0  # Where the red mask is applied
    image_array[overlay] = (
        image_array[overlay] * 0.5 + red_mask_rgba_array[overlay] * 0.5
    )  # Blend image and mask

    # Convert back to Image for saving
    final_image = Image.fromarray(image_array)

    # Show the result
    plt.figure(figsize=(10, 10))
    plt.imshow(final_image)
    plt.axis("off")
    plt.show()

    # Optionally save the resulting image
    if output_path:
        final_image.save(output_path)


# Example usage
image_path = "./atlantis/images/106416826.jpg"
mask_path = "./atlantis/processed_masks/106416826.png"

overlay_mask_on_image(image_path, mask_path)
