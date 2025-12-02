"""
============================================================
MULTI-MODEL SEGMENTATION PIPELINE
============================================================
Pipeline che testa diversi modelli di segmentazione:
- BasicUNet (from scratch)
- SMP UNet (ResNet50)
- SMP MAnet (MiT-B0)
- SMP Segformer (MiT-B0)
- SMP DeepLabV3+ (ResNet50)
- Attention U-Net (custom implementation)

Ogni modello viene trainato, valutato e poi la memoria viene liberata.
============================================================
"""

import os
import gc
import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import models
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from sklearn.model_selection import train_test_split

from tqdm import tqdm

# NOTE: segmentation_models_pytorch is required for SMP models
# Install with: pip install segmentation-models-pytorch

# ============================================================
# CONFIGURATION
# ============================================================

# Paths - MODIFY THESE FOR YOUR ENVIRONMENT
DATA_DIR = "/kaggle/input/tumor-segmentation-ai/training_images/training_images"
EXCEL_PATH = "/kaggle/input/tumor-segmentation-ai/training_metadata.xlsx"
OUTPUT_DIR = "/kaggle/working/"

# Hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS_CLASSIFICATION = 50
NUM_EPOCHS_SEGMENTATION = 50
LR_CLASSIFICATION = 1e-3
LR_SEGMENTATION = 2e-4
EARLY_STOPPING_PATIENCE = 15

# Class mapping
id2class = {0: "benigno", 1: "maligno", 2: "sano"}
class2id = {v: k for k, v in id2class.items()}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def cleanup_gpu():
    """Free GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")


# ============================================================
# DATASET CLASSES
# ============================================================

class UltrasoundDataset(Dataset):
    def __init__(self, metadata_df, images_dir, mode='segmentation', transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.images_dir = images_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['US']
        mask_name = row.get('MASK', None)
        label = row.get('LABEL', None)

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.mode == 'segmentation':
            if mask_name is None:
                raise ValueError("MASK mancante per modalità 'segmentation'")
            mask_path = os.path.join(self.images_dir, mask_name)
            mask = Image.open(mask_path).convert("L")
            return image, mask

        elif self.mode == 'classification':
            if self.transform:
                image = self.transform(image)
            return image, int(label)

        else:
            raise ValueError(f"mode sconosciuto: {self.mode}")


# Joint transforms for segmentation
class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class JointResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        return (
            TF.resize(img, self.size),
            TF.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST),
        )


class JointToTensor:
    def __call__(self, img, mask):
        return TF.to_tensor(img), TF.to_tensor(mask)


class JointRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return TF.hflip(img), TF.hflip(mask)
        return img, mask


class JointRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask


class JointRandomRotation:
    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, img, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)
        return img, mask


class JointNormalize:
    def __init__(self, mean, std):
        self.norm = T.Normalize(mean=mean, std=std)

    def __call__(self, img, mask):
        return self.norm(img), mask


class SegmentationWrapper(Dataset):
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        img, mask = self.transform(img, mask)
        return img, mask


def get_dataloaders(data_dir, excel_path, batch_size=16, mode='segmentation', img_size=256):
    """Create train/val/test dataloaders"""
    
    if not os.path.exists(data_dir) or not os.path.exists(excel_path):
        raise FileNotFoundError(f"Data not found at {data_dir} or {excel_path}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    df = pd.read_excel(excel_path)
    train_df, test_val_df = train_test_split(df, test_size=0.3, stratify=df['LABEL'], random_state=42)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df['LABEL'], random_state=42)

    if mode == 'segmentation':
        train_base = UltrasoundDataset(train_df, data_dir, mode='segmentation')
        val_base = UltrasoundDataset(val_df, data_dir, mode='segmentation')
        test_base = UltrasoundDataset(test_df, data_dir, mode='segmentation')

        train_transform = JointCompose([
            JointResize((img_size, img_size)),
            JointRandomHorizontalFlip(p=0.5),
            JointRandomVerticalFlip(p=0.5),
            JointRandomRotation(degrees=10),
            JointToTensor(),
            JointNormalize(mean, std),
        ])

        val_transform = JointCompose([
            JointResize((img_size, img_size)),
            JointToTensor(),
            JointNormalize(mean, std),
        ])

        train_ds = SegmentationWrapper(train_base, train_transform)
        val_ds = SegmentationWrapper(val_base, val_transform)
        test_ds = SegmentationWrapper(test_base, val_transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        return train_loader, val_loader, test_loader

    # Classification mode
    normalize_tf = T.Normalize(mean=mean, std=std)
    
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        normalize_tf,
    ])

    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        normalize_tf,
    ])

    train_ds = UltrasoundDataset(train_df, data_dir, mode='classification', transform=train_transform)
    val_ds = UltrasoundDataset(val_df, data_dir, mode='classification', transform=val_transform)
    test_ds = UltrasoundDataset(test_df, data_dir, mode='classification', transform=val_transform)

    labels = train_df['LABEL'].values
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = {cls: 1.0 / c for cls, c in zip(classes, counts)}
    sample_weights = np.array([class_weights[l] for l in labels], dtype=np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


# ============================================================
# SEGMENTATION MODEL DEFINITIONS
# ============================================================

# 1) BasicUNet from scratch
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class BasicUNet(nn.Module):
    """Basic U-Net implementation from scratch"""
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.down1 = DoubleConv(n_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.middle = DoubleConv(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        self.returns_logits = False  # Returns sigmoid output

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        mid = self.middle(p2)
        u2 = self.up2(mid)
        cat2 = torch.cat([u2, c2], dim=1)
        c3 = self.conv2(cat2)
        u1 = self.up1(c3)
        cat1 = torch.cat([u1, c1], dim=1)
        c4 = self.conv1(cat1)
        out = self.out_conv(c4)
        return torch.sigmoid(out)


# 2) Attention U-Net (converted from Keras to PyTorch)
class AttentionGate(nn.Module):
    """Attention Gate for U-Net"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EncoderBlock(nn.Module):
    """Encoder block with optional pooling"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, pooling=True):
        super().__init__()
        self.pooling = pooling
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if pooling:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c = self.conv(x)
        if self.pooling:
            p = self.pool(c)
            return p, c
        return c


class DecoderBlock(nn.Module):
    """Decoder block with upsampling"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class AttentionUNet(nn.Module):
    """Attention U-Net for segmentation"""
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(n_channels, 32, 0.1)
        self.enc2 = EncoderBlock(32, 64, 0.1)
        self.enc3 = EncoderBlock(64, 128, 0.2)
        self.enc4 = EncoderBlock(128, 256, 0.2)
        
        # Bottleneck
        self.bottleneck = EncoderBlock(256, 512, 0.3, pooling=False)
        
        # Attention gates
        self.att4 = AttentionGate(512, 256, 256)
        self.att3 = AttentionGate(256, 128, 128)
        self.att2 = AttentionGate(128, 64, 64)
        self.att1 = AttentionGate(64, 32, 32)
        
        # Decoder
        self.dec4 = DecoderBlock(512, 256, 0.2)
        self.dec3 = DecoderBlock(256, 128, 0.2)
        self.dec2 = DecoderBlock(128, 64, 0.1)
        self.dec1 = DecoderBlock(64, 32, 0.1)
        
        # Output
        self.out_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        self.returns_logits = False  # Returns sigmoid output

    def forward(self, x):
        # Encoder
        p1, c1 = self.enc1(x)
        p2, c2 = self.enc2(p1)
        p3, c3 = self.enc3(p2)
        p4, c4 = self.enc4(p3)
        
        # Bottleneck
        bn = self.bottleneck(p4)
        
        # Decoder with attention
        a4 = self.att4(bn, c4)
        d4 = self.dec4(bn, a4)
        
        a3 = self.att3(d4, c3)
        d3 = self.dec3(d4, a3)
        
        a2 = self.att2(d3, c2)
        d2 = self.dec2(d3, a2)
        
        a1 = self.att1(d2, c1)
        d1 = self.dec1(d2, a1)
        
        out = self.out_conv(d1)
        return torch.sigmoid(out)


# ============================================================
# SEGMENTATION METRICS AND LOSS
# ============================================================

def dice_coeff(pred, target, eps=1e-6):
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    den = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (den + eps)
    return dice.mean()


def iou_coeff(pred, target, eps=1e-6):
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def estimate_pos_weight(loader, max_batches=10):
    pos, neg = 0, 0
    with torch.no_grad():
        for i, (_, masks) in enumerate(loader):
            m = (masks > 0.5)
            pos += m.sum().item()
            neg += (~m).sum().item()
            if i + 1 >= max_batches:
                break
    return max(1.0, neg / pos) if pos > 0 else 1.0


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss for segmentation"""
    def __init__(self, pos_weight=1.0, bce_weight=0.2, dice_weight=0.8):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target, use_sigmoid=False):
        if use_sigmoid:
            # Model already applied sigmoid
            pred_logits = torch.logit(pred.clamp(1e-7, 1-1e-7))
            bce_loss = self.bce(pred_logits, target)
            dice_loss = 1.0 - dice_coeff(pred, target)
        else:
            bce_loss = self.bce(pred, target)
            probs = torch.sigmoid(pred)
            dice_loss = 1.0 - dice_coeff(probs, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ============================================================
# MODEL FACTORY
# ============================================================

def create_segmentation_model(model_name, in_channels=3, classes=1):
    """Factory function to create segmentation models"""
    
    if model_name == 'BasicUNet':
        model = BasicUNet(n_channels=in_channels, n_classes=classes)
        model.returns_logits = False
        return model
    
    elif model_name == 'AttentionUNet':
        model = AttentionUNet(n_channels=in_channels, n_classes=classes)
        model.returns_logits = False
        return model
    
    # Import SMP models
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation_models_pytorch is required for SMP models. "
            "Please install it with: pip install segmentation-models-pytorch"
        )
    
    if model_name == 'SMP_UNet_ResNet50':
        model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", 
                        in_channels=in_channels, classes=classes)
        model.returns_logits = True
        return model
    
    elif model_name == 'SMP_MAnet_MiTB0':
        model = smp.MAnet(encoder_name="mit_b0", encoder_weights="imagenet",
                         in_channels=in_channels, classes=classes)
        model.returns_logits = True
        return model
    
    elif model_name == 'SMP_Segformer_MiTB0':
        model = smp.Segformer(encoder_name="mit_b0", encoder_weights="imagenet",
                             in_channels=in_channels, classes=classes)
        model.returns_logits = True
        return model
    
    elif model_name == 'SMP_DeepLabV3Plus_ResNet50':
        model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet",
                                  in_channels=in_channels, classes=classes)
        model.returns_logits = True
        return model
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_segmentation_model(model, train_loader, val_loader, model_name, 
                            num_epochs=50, lr=2e-4, device='cuda'):
    """Train a single segmentation model"""
    
    print_section(f"Training: {model_name}")
    
    model = model.to(device)
    returns_logits = getattr(model, 'returns_logits', True)
    
    # Loss
    pos_weight = estimate_pos_weight(train_loader)
    print(f"Estimated pos_weight: {pos_weight:.2f}")
    criterion = CombinedLoss(pos_weight=pos_weight)
    criterion.bce = criterion.bce.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    
    best_val_dice = 0.0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss, train_dice = 0.0, 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            y = (masks > 0.5).float()
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, y, use_sigmoid=not returns_logits)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                probs = outputs if not returns_logits else torch.sigmoid(outputs)
                dice = dice_coeff(probs, y)
            
            train_loss += loss.item() * imgs.size(0)
            train_dice += dice.item() * imgs.size(0)
            pbar.set_postfix({'loss': loss.item(), 'dice': dice.item()})
        
        train_loss /= len(train_loader.dataset)
        train_dice /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                y = (masks > 0.5).float()
                outputs = model(imgs)
                loss = criterion(outputs, y, use_sigmoid=not returns_logits)
                probs = outputs if not returns_logits else torch.sigmoid(outputs)
                
                val_loss += loss.item() * imgs.size(0)
                val_dice += dice_coeff(probs, y).item() * imgs.size(0)
                val_iou += iou_coeff(probs, y).item() * imgs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        
        scheduler.step(val_dice)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_dice={train_dice:.4f}, "
              f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}, val_iou={val_iou:.4f}")
        
        if val_dice > best_val_dice + 1e-4:
            best_val_dice = val_dice
            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best! Val Dice: {best_val_dice:.4f}")
    
    print(f"\n{model_name} - Best validation Dice: {best_val_dice:.4f}")
    return best_state, history, best_val_dice


def evaluate_segmentation_model(model, test_loader, model_name, device='cuda'):
    """Evaluate a segmentation model on test set"""
    
    model = model.to(device)
    model.eval()
    returns_logits = getattr(model, 'returns_logits', True)
    
    test_dice, test_iou, n_samples = 0.0, 0.0, 0
    
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc=f"Testing {model_name}"):
            imgs, masks = imgs.to(device), masks.to(device)
            y = (masks > 0.5).float()
            outputs = model(imgs)
            probs = outputs if not returns_logits else torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            b = imgs.size(0)
            test_dice += dice_coeff(preds, y).item() * b
            test_iou += iou_coeff(preds, y).item() * b
            n_samples += b
    
    test_dice /= n_samples
    test_iou /= n_samples
    
    print(f"\n{model_name} Test Results: Dice={test_dice:.4f}, IoU={test_iou:.4f}")
    return {"model_name": model_name, "test_dice": test_dice, "test_iou": test_iou}


# ============================================================
# CLASSIFICATION PIPELINE
# ============================================================

def train_classification_model(train_loader, val_loader, num_classes=3, 
                               num_epochs=50, lr=1e-3, device='cuda'):
    """Train classification model (ResNet50)"""
    
    print_section("Training Classification Model (ResNet50)")
    
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, num_classes),
    )
    
    # Unfreeze layer3 and fc
    for name, param in model.named_parameters():
        if name.startswith("layer3") or name.startswith("fc"):
            param.requires_grad = True
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best! Val Acc: {best_val_acc:.4f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def compute_predictions_for_all(model, excel_path, data_dir, img_size=256, device='cuda'):
    """Compute predictions for all images to create segmentation dataset"""
    
    df = pd.read_excel(excel_path)
    
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    preds = []
    model.eval()
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            img_path = os.path.join(data_dir, row["US"])
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            out = model(img)
            pred = out.argmax(dim=1).item()
            preds.append(pred)
    
    df["PRED_LABEL"] = preds
    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():
    """Main pipeline execution"""
    
    print_section("MULTI-MODEL SEGMENTATION PIPELINE")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ========== PHASE 1: CLASSIFICATION ==========
    print_section("PHASE 1: CLASSIFICATION")
    
    train_loader_clf, val_loader_clf, test_loader_clf = get_dataloaders(
        data_dir=DATA_DIR,
        excel_path=EXCEL_PATH,
        batch_size=BATCH_SIZE,
        mode="classification",
        img_size=IMG_SIZE
    )
    
    print(f"Train batches: {len(train_loader_clf)}")
    print(f"Val batches: {len(val_loader_clf)}")
    print(f"Test batches: {len(test_loader_clf)}")
    
    clf_model = train_classification_model(
        train_loader_clf, val_loader_clf,
        num_classes=3,
        num_epochs=NUM_EPOCHS_CLASSIFICATION,
        lr=LR_CLASSIFICATION,
        device=device
    )
    
    # Generate predictions for segmentation dataset
    pred_df = compute_predictions_for_all(clf_model, EXCEL_PATH, DATA_DIR, IMG_SIZE, device)
    pred_df["LABEL"] = pred_df["PRED_LABEL"]
    seg_df = pred_df[pred_df["LABEL"] != 2].reset_index(drop=True)
    print(f"Segmentation dataset size: {len(seg_df)}")
    
    seg_metadata_path = os.path.join(OUTPUT_DIR, "temp_segmentation_metadata.xlsx")
    seg_df.to_excel(seg_metadata_path, index=False)
    
    # Clean up classification model
    del clf_model
    cleanup_gpu()
    
    # ========== PHASE 2: SEGMENTATION ==========
    print_section("PHASE 2: SEGMENTATION - TESTING MULTIPLE MODELS")
    
    train_loader_seg, val_loader_seg, test_loader_seg = get_dataloaders(
        data_dir=DATA_DIR,
        excel_path=seg_metadata_path,
        batch_size=BATCH_SIZE,
        mode="segmentation",
        img_size=IMG_SIZE
    )
    
    print(f"Train batches (seg): {len(train_loader_seg)}")
    print(f"Val batches (seg): {len(val_loader_seg)}")
    print(f"Test batches (seg): {len(test_loader_seg)}")
    
    # Models to test
    MODELS_TO_TEST = [
        'BasicUNet',
        'AttentionUNet',
        'SMP_UNet_ResNet50',
        'SMP_MAnet_MiTB0',
        'SMP_Segformer_MiTB0',
        'SMP_DeepLabV3Plus_ResNet50',
    ]
    
    all_results = []
    all_histories = {}
    
    for model_name in MODELS_TO_TEST:
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*80}")
        
        try:
            # Create model
            model = create_segmentation_model(model_name, in_channels=3, classes=1)
            
            # Train
            best_state, history, best_val_dice = train_segmentation_model(
                model=model,
                train_loader=train_loader_seg,
                val_loader=val_loader_seg,
                model_name=model_name,
                num_epochs=NUM_EPOCHS_SEGMENTATION,
                lr=LR_SEGMENTATION,
                device=device
            )
            
            # Load best weights
            if best_state is not None:
                model.load_state_dict(best_state)
            
            # Evaluate
            results = evaluate_segmentation_model(model, test_loader_seg, model_name, device)
            results['best_val_dice'] = best_val_dice
            
            all_results.append(results)
            all_histories[model_name] = history
            
            # Save model
            save_path = os.path.join(OUTPUT_DIR, f"best_{model_name}.pth")
            torch.save(best_state, save_path)
            print(f"Model saved to: {save_path}")
            
        except Exception as e:
            print(f"ERROR training {model_name}: {e}")
            all_results.append({"model_name": model_name, "test_dice": 0, "test_iou": 0, "error": str(e)})
        
        finally:
            # IMPORTANT: Clean up memory after each model
            if 'model' in locals():
                del model
            if 'best_state' in locals():
                del best_state
            cleanup_gpu()
            print(f"Memory cleaned after {model_name}")
    
    # ========== SUMMARY ==========
    print_section("FINAL RESULTS SUMMARY")
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('test_dice', ascending=False)
    print(results_df.to_string(index=False))
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, "segmentation_models_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Plot training curves
    if all_histories:
        n_models = len(all_histories)
        fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 8))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for idx, (model_name, history) in enumerate(all_histories.items()):
            epochs = range(1, len(history["train_loss"]) + 1)
            
            axes[0, idx].plot(epochs, history["train_loss"], label="Train")
            axes[0, idx].plot(epochs, history["val_loss"], label="Val")
            axes[0, idx].set_title(f"{model_name}\nLoss")
            axes[0, idx].legend()
            axes[0, idx].grid(True)
            
            axes[1, idx].plot(epochs, history["train_dice"], label="Train")
            axes[1, idx].plot(epochs, history["val_dice"], label="Val")
            axes[1, idx].set_title(f"{model_name}\nDice")
            axes[1, idx].legend()
            axes[1, idx].grid(True)
        
        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, "training_curves.png")
        plt.savefig(fig_path, dpi=150)
        plt.show()
        print(f"Training curves saved to: {fig_path}")
    
    print("\n" + "="*80)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return results_df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    results = run_pipeline()
