import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import numpy as np
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import torch
import torchvision.transforms as transforms
from PIL import Image

sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
sam_model_type = "vit_h"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
sam.to(device)
sam.eval()
mask_generator = SamAutomaticMaskGenerator(sam)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_size = 640
batch_size = 4
num_epochs = 5
learning_rate = 1e-4
image_dir = './datasets/FlowIMG_FloatingWaste/images/train'
annotation_dir = './datasets/FlowIMG_FloatingWaste/labels/train'
save_model_path = 'unet.pth'
unet_output_dir = 'unet_outputs'


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hook_handles = []
    
    def hook_layer(self, layer_name, layer):
        def hook(module, input, output):
            self.features[layer_name] = output
        return layer.register_forward_hook(hook)
    
    def extract_features(self, image):
        self.features = {} 
        with torch.no_grad():
            image_embeddings = self.model.image_encoder(image)
            
        return self.features


def preprocess_image(image_path, input_size=1024, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def setup_sam_feature_extraction(sam, device='cuda'):
    feature_extractor = FeatureExtractor(sam)
    
    layers_to_hook = {
        "patch_embedding": sam.image_encoder.patch_embed.proj,
        "neck": sam.image_encoder.neck,
        "pre_final": sam.mask_decoder.transformer.final_attn_token_to_image,
    }
    
    hook_handles = []
    for name, layer in layers_to_hook.items():
        hook_handles.append(feature_extractor.hook_layer(name, layer))
    
    return feature_extractor, hook_handles

def extract_sam_features(sam, input_image, device='cuda'):
    feature_extractor, hook_handles = setup_sam_feature_extraction(sam, device)

    # Ensure hooks are removed after forward pass
    try:
        features = feature_extractor.extract_features(input_image)
    finally:
        pass

    return features

class FeatureProcessor(nn.Module):
    def __init__(self, feature_channels, output_channels, input_size=640):
        super(FeatureProcessor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, input_channels, output_channels=3):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        out = self.final_conv(dec1)
        out = torch.sigmoid(out)  # Assuming output image normalized between 0 and 1
        return out

def train_unet(model, dataloader, optimizer, criterion, device, num_epochs=10, save_path='unet.pth'):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (features, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            features = features.to(device).float()
            targets = targets.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Save the model checkpoint after each epoch
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn

# Assuming UNet, FeatureProcessor, setup_sam_feature_extraction, extract_sam_features, preprocess_image, train_unet are defined elsewhere

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, sam, device='cuda', input_size=640, include_boxes=False):
        """
        Args:
            image_dir (str): Path to the directory containing input images.
            annotation_dir (str): Path to the directory containing YOLO format annotation files.
            sam (nn.Module): The SAM model for feature extraction.
            device (str): Device to load tensors onto ('cuda' or 'cpu').
            input_size (int): Size to which images are resized.
            include_boxes (bool): Whether to include bounding boxes in the returned data.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_paths = sorted([
            os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.annotation_paths = sorted([
            os.path.join(annotation_dir, fname) for fname in os.listdir(annotation_dir)
            if fname.lower().endswith('.txt')
        ])
        assert len(self.image_paths) == len(self.annotation_paths), "Mismatch between images and annotations"
        
        self.sam = sam
        self.device = device
        self.input_size = input_size
        self.include_boxes = include_boxes
        
        # Initialize feature extractor and hooks
        self.feature_extractor, self.hook_handles = setup_sam_feature_extraction(self.sam, device)
        
        # Initialize feature processors for each layer
        # Adjust feature_channels based on actual SAM model's output channels
        self.feature_processors = nn.ModuleDict({
            "patch_embedding": FeatureProcessor(feature_channels=1280, output_channels=64, input_size=input_size),
            "neck": FeatureProcessor(feature_channels=256, output_channels=128, input_size=input_size),
            "pre_final": FeatureProcessor(feature_channels=256, output_channels=128, input_size=input_size),
        }).to(device)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)  # Extract the filename
        
        features = extract_sam_features(self.sam, preprocess_image(image_path))
        image = preprocess_image(image_path, input_size=self.input_size, device=self.device)
        
        # Process each feature map
        processed_features = []
        for layer_name, feature_map in features.items():
            processed = self.feature_processors[layer_name](feature_map)
            processed_features.append(processed)
        
        # Concatenate along channel dimension
        concatenated_features = torch.cat(processed_features, dim=1)  # Shape: [batch, channels, H, W]
        
        # Get target image (original image normalized between 0 and 1)
        target_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
        ])
        target_image = target_transform(Image.open(image_path).convert("RGB")).to(self.device)
        
        if self.include_boxes:
            # Load annotation (YOLO format)
            annotation_path = self.annotation_paths[idx]
            boxes = self.load_annotation(annotation_path)
            return concatenated_features.squeeze(0).cpu(), target_image.squeeze(0).cpu(), boxes, filename
        else:
            return concatenated_features.squeeze(0).cpu(), target_image.squeeze(0).cpu(), filename
    
    def load_annotation(self, annotation_path):
        # Load YOLO format bounding boxes
        # Each line in annotation file: class x_center y_center width height (all normalized)
        boxes = []
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_center, y_center, width, height = map(float, parts)
                boxes.append([cls, x_center, y_center, width, height])
        return boxes
    
    def __del__(self):
        # Remove hooks
        for handle in self.hook_handles:
            handle.remove()

def generate_unet_outputs(model, dataloader, device, output_dir='unet_outputs'):
    """
    Generates output images using the U-Net model and saves them with the same filenames as the input images.
    
    Args:
        model (nn.Module): The trained U-Net model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
        output_dir (str): Directory to save the output images.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Output Images"):
            if dataset.include_boxes:
                features, _, _, filenames = batch
            else:
                features, _, filenames = batch
            
            features = features.to(device).float()
            outputs = model(features)
            outputs = outputs.cpu().numpy()
            batch_size = outputs.shape[0]
            
            for i in range(batch_size):
                output_img = np.transpose(outputs[i], (1, 2, 0))  # CxHxW to HxWxC
                
                # Depending on the model's output activation, adjust scaling
                # For example, if output is in [0,1], scale to [0,255]
                output_img = np.clip(output_img, 0, 1)  # Ensure values are within [0,1]
                output_img = (output_img * 255).astype(np.uint8)
                
                # If the output has a single channel, convert to 3-channel image
                if output_img.shape[2] == 1:
                    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
                else:
                    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                
                # Save the image with the same filename as the input
                save_path = os.path.join(output_dir, filenames[i])
                cv2.imwrite(save_path, output_img)
    
    print(f"U-Net output images saved to '{output_dir}' directory.")

# Training Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_size = 640
batch_size = 4
num_epochs = 20
learning_rate = 1e-4
image_dir = 'datasets/FlowIMG_FloatingWaste/images/val'
annotation_dir = 'datasets/FlowIMG_FloatingWaste/labels/val'
save_model_path = 'unet.pth'
unet_output_dir = 'datasets/FlowIMG_FloatingWaste/images/val_unet'

# Initialize Dataset and DataLoader
dataset = CustomDataset(
    image_dir=image_dir,
    annotation_dir=annotation_dir,
    sam=sam,
    device=device,
    input_size=input_size,
    include_boxes=False  # Set to True if you want to include bounding boxes
)

# Define a custom collate_fn to handle the additional filename
def custom_collate(batch):
    if dataset.include_boxes:
        features, target_images, boxes, filenames = zip(*batch)
        return torch.stack(features, 0), torch.stack(target_images, 0), boxes, filenames
    else:
        features, target_images, filenames = zip(*batch)
        return torch.stack(features, 0), torch.stack(target_images, 0), filenames

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=custom_collate
)

input_channels = 64 + 128  # Adjust if necessary
unet = UNet(input_channels=input_channels).to(device)
save_model_path = 'unet.pth'  # Path to the .pth file
unet.load_state_dict(torch.load(save_model_path, map_location=device))

criterion = nn.L1Loss()  # Mean Absolute Error
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

#train_unet(unet, dataloader, optimizer, criterion, device, num_epochs=num_epochs, save_path=save_model_path)

generate_unet_outputs(unet, dataloader, device, output_dir=unet_output_dir)

print("U-Net training and output generation completed.")

