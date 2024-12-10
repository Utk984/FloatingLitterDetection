import os
import numpy as np
from tqdm import tqdm
from glob import glob
from numpy import zeros
from numpy.random import randint
import torch
import cv2
from statistics import mean
from torch.nn.functional import threshold, normalize
import matplotlib.pyplot as plt
from pathlib import Path
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from statistics import mean
from collections import defaultdict
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import multiprocessing
import logging

logging.basicConfig(filename='training_waterv2.log', level=logging.INFO, format='%(asctime)s - %(message)s')

lr = 1e-5
wd = 0
batch_size = 4
num_epochs = 50
desired_size = (512, 512)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_type = 'vit_h'
checkpoint = 'models/sam_vit_h_4b8939.pth'
sam_model_combined = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model_combined.to(device)
sam_model_combined.train()
#sam_model_focal = sam_model_registry[model_type](checkpoint=checkpoint)
#sam_model_focal.to(device)
#sam_model_focal.train() 


#path = "/home/hpc/visitor/px019.visitor/datasets/LuFI-RiverSnap.v1/"
#image_path = f"{path}Train/Images"
#label_path = f"{path}Train/Labels"
#val_image_path = f"{path}Val/Images"
#val_label_path = f"{path}Val/Labels"

#train_image_paths = sorted(glob(image_path + "/*.jpg"))
#train_label_paths = sorted(glob(label_path + "/*.png"))
#val_image_paths = sorted(glob(val_image_path + "/*.jpg"))
#val_label_paths = sorted(glob(val_label_path + "/*.png"))

path2 = "/home/hpc/visitor/px019.visitor/datasets/water_v2/water_v2/"
image_path = f"{path2}JPEGImages/ADE20K"
label_path = f"{path2}Annotations/ADE20K"
val_image_path = f"{path2}JPEGImages/river_segs"
val_label_path = f"{path2}Annotations/river_segs"

train_image_paths = sorted(glob(image_path + "/*.png"))
train_label_paths = sorted(glob(label_path + "/*.png"))
val_image_paths = sorted(glob(val_image_path + "/*.jpg"))
val_label_paths = sorted(glob(val_label_path + "/*.png"))

logging.info(len(train_image_paths))
logging.info(len(train_label_paths))

# Read training masks
ground_truth_masks = {}
for k in range(len(train_label_paths)):
    gt_grayscale = cv2.imread(train_label_paths[k], cv2.IMREAD_GRAYSCALE)
    if desired_size is not None:
        gt_grayscale = cv2.resize(gt_grayscale, desired_size, interpolation=cv2.INTER_LINEAR)
    ground_truth_masks[k] = (gt_grayscale > 0)

# Read validation masks
ground_truth_masksv = {}
for s in range(len(val_label_paths)):
    gt_grayscale = cv2.imread(val_label_paths[s], cv2.IMREAD_GRAYSCALE)
    if desired_size is not None:
        gt_grayscale = cv2.resize(gt_grayscale, desired_size, interpolation=cv2.INTER_LINEAR)
    ground_truth_masksv[s] = (gt_grayscale > 0)

keys = list(ground_truth_masks.keys())
keys1 = list(ground_truth_masksv.keys())

transformed_data = defaultdict(dict)
for k in range(len(train_image_paths)):
    image = cv2.imread(train_image_paths[k])
    if desired_size is not None:
        image = cv2.resize(image, desired_size, interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = ResizeLongestSide(sam_model_combined.image_encoder.img_size)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_image = sam_model_combined.preprocess(transformed_image)
    original_image_size = image.shape[:2]
    input_size = tuple(transformed_image.shape[-2:])

    transformed_data[k]['image'] = input_image
    transformed_data[k]['input_size'] = input_size
    transformed_data[k]['original_image_size'] = original_image_size

del train_image_paths
del train_label_paths

def dice_loss(predictions, targets):
    """
    Dice Loss: Measures overlap between predicted and true labels.
    """
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum()
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return 1 - dice  # To minimize the loss

def combined_loss(predictions, targets):
    """
    Combined BCE and Dice Loss.
    BCE with logits loss is used when the output layer has no activation function (logits).
    """
    bce_loss = nn.BCEWithLogitsLoss()(predictions, targets)
    dice_loss_val = dice_loss(predictions, targets)
    return bce_loss + dice_loss_val

def focal_loss(predictions, targets):
    """
    Focal Loss: Addresses class imbalance by focusing more on hard-to-classify examples.
    """
    predictions = torch.sigmoid(predictions)
    cross_entropy = -targets * torch.log(predictions + 1e-6) - (1 - targets) * torch.log(1 - predictions + 1e-6)
    pt = torch.exp(-cross_entropy)  # Probability of the correct class
    focal_weight = 0.25 * (1 - pt) ** 2.0  # The modulating factor
    focal_loss = focal_weight * cross_entropy
    return focal_loss.mean()

def calculate_accuracy(predictions, targets):
    binary_predictions = (predictions > 0.5).float()
    accuracy = (binary_predictions == targets).float().mean()
    return accuracy.item()

def calculate_iou(predictions, targets, smooth=1e-6):
    binary_predictions = (predictions > 0.5).float()  # Assuming predictions are logits or probabilities
    binary_targets = (targets > 0.5).float()
    intersection = (binary_predictions * binary_targets).sum()
    union = (binary_predictions + binary_targets).sum()  # Element-wise addition for union
    iou = (intersection + smooth) / (union + smooth)  # Add small epsilon to avoid division by zero
    return iou.item()

def calculate_dice_score(predictions, targets, smooth=1e-6):
    binary_predictions = (predictions > 0.5).float()  # Assuming predictions are logits or probabilities
    binary_targets = (targets > 0.5).float()
    intersection = (binary_predictions * binary_targets).sum()
    union = binary_predictions.sum() + binary_targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def log_metrics(model_name, epoch, loss, accuracy, iou, dice, val_loss, val_accuracy):
    logging.info(f"Model {model_name} | Epoch {epoch} | Train Loss: {loss:.4f} | Train Accuracy: {accuracy:.4f} | Train IoU: {iou:.4f} | Train Dice: {dice:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

def train_on_batch(keys, batch_start, batch_end, loss_fn, optimizer, sam_model):
    batch_losses = []
    batch_accuracies = []
    batch_iou = []
    batch_dice = []

    for k in keys[batch_start:batch_end]:
        input_image = transformed_data[k]['image'].to(device)
        input_size = transformed_data[k]['input_size']
        original_image_size = transformed_data[k]['original_image_size']

        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        binary_mask = (threshold(torch.sigmoid(upscaled_masks), 0.5, 0))
        gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
        gt_mask_resized = gt_mask_resized > 0.5
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

        loss = loss_fn(binary_mask, gt_binary_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

        # Calculate accuracy for training data
        train_accuracy = calculate_accuracy(binary_mask, gt_binary_mask)
        train_dice = calculate_dice_score(binary_mask, gt_binary_mask)
        train_iou = calculate_iou(binary_mask, gt_binary_mask)
        batch_accuracies.append(train_accuracy)
        batch_iou.append(train_iou)
        batch_dice.append(train_dice)

    return batch_losses, batch_accuracies, batch_iou, batch_dice

def validate(predictor_tuned, loss_fn):
    val_loss = 0.0
    val_accuracy = 0.0
    num_val_examples = 0
    with torch.no_grad():
        for s in keys1[:len(val_image_paths)]:
            image = cv2.imread(val_image_paths[s])
            if desired_size is not None:
                image = cv2.resize(image, desired_size, interpolation=cv2.INTER_LINEAR)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Forward pass on validation data
            predictor_tuned.set_image(image)

            masks_tuned, _, _ = predictor_tuned.predict(
                point_coords=None,
                box=None,
                multimask_output=False,
            )

            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masksv[s], (1, 1, ground_truth_masksv[s].shape[0], ground_truth_masksv[s].shape[1]))).to(device)
            gt_mask_resized = gt_mask_resized > 0.5
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
            masks_tuned1 = torch.as_tensor(masks_tuned > 0, dtype=torch.float32)
            new_tensor = masks_tuned1.unsqueeze(0).to(device)

            # Calculate validation loss
            val_loss += loss_fn(new_tensor, gt_binary_mask).item()

            # Calculate accuracy for validation data
            val_accuracy += calculate_accuracy(new_tensor, gt_binary_mask)
            num_val_examples += 1
    return val_loss, val_accuracy, num_val_examples

def train_model(sam_model, model_name, loss_fn, optimizer):
    losses = []
    val_losses = []
    accuracies = []
    ious = []
    dices = []
    best_val_loss = float('inf')  
    val_acc = []
    no_improve_count = 0 
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_accuracies = []
        epoch_ious = []
        epoch_dices = []
    
        # Training loop with batch processing
        for batch_start in range(0, len(keys), batch_size):
            torch.cuda.empty_cache()
            batch_end = min(batch_start + batch_size, len(keys))
    
            batch_losses, batch_accuracies, batch_ious, batch_dices = train_on_batch(keys, batch_start, batch_end, loss_fn, optimizer, sam_model)
            epoch_accuracies.extend(batch_accuracies)
            epoch_ious.extend(batch_ious)
            epoch_dices.extend(batch_dices)
    
            # Calculate mean training loss for the current batch
            batch_loss = mean(batch_losses)
            epoch_losses.append(batch_loss)
    
        # Calculate mean training loss for the current epoch
        losses.append(mean(epoch_losses))
        accuracies.append(mean(epoch_accuracies))
        ious.append(mean(epoch_ious))
        dices.append(mean(epoch_dices))
    
        predictor_tuned = SamPredictor(sam_model)
        val_loss, val_accuracy, num_val_examples = validate(predictor_tuned, loss_fn)
    
        # Calculate mean validation loss for the current epoch
        val_loss /= num_val_examples
        val_losses.append(val_loss)
    
        # Calculate mean validation accuracy for the current epoch
        mean_val_accuracy = val_accuracy / num_val_examples
        val_acc.append(mean_val_accuracy)
    
        print(f'Mean validation loss: {val_loss}')
        print(f'Mean validation accuracy: {mean_val_accuracy}')
    
        log_metrics(model_name, epoch, mean(epoch_losses), mean(epoch_accuracies), mean(epoch_ious), mean(epoch_dices), val_loss, mean_val_accuracy)

        # Save the model checkpoint if the validation accuracy improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0 
            models_path = 'final_models'
            torch.save(sam_model.state_dict(), os.path.join(models_path, f'{model_name}.pth'))
        else:
            no_improve_count += 1
    
        if no_improve_count >= 4:
            break
 
        torch.cuda.empty_cache()
    
    return losses, accuracies, ious, dices, val_acc, val_losses

def train_combined_model(model_name):
    loss_fn_combined = combined_loss
    optimizer_combined = torch.optim.Adam(sam_model_combined.mask_decoder.parameters(), lr=lr, weight_decay=wd)
    losses_c, accuracies_c, ious_c, dices_c, val_acc_c, val_losses_c = train_model(sam_model_combined, model_name, loss_fn_combined, optimizer_combined)
    return losses_c, accuracies_c, ious_c, dices_c, val_acc_c, val_losses_c

def plot_comparison(x, y1, y2, title, xlabel, ylabel, label1, label2, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_path)


if __name__ == '__main__':
    # Start the processes for parallel training
    # losses_c, accuracies_c, ious_c, dices_c, val_acc_c, val_losses_c = train_combined_model()
    losses_f, accuracies_f, ious_f, dices_f, val_acc_f, val_losses_f = train_combined_model("sam_model_waterv2")
