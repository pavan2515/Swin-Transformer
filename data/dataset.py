"""
Dataset implementation for handwritten character recognition
"""
import os
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import sys
sys.path.append('..')
from utils.preprocessing import preprocess_manuscript_image


class HandwrittenDataset(Dataset):
    """
    Dataset for handwritten character images organized in folders by label
    
    Expected folder structure:
    root_dir/
        ├── a/
        │   ├── img1.png
        │   ├── img2.png
        ├── b/
        │   ├── img1.png
        ...
    """
    
    def __init__(
        self,
        root_dir: str,
        tokenizer,
        image_size: int = 224,
        apply_preprocessing: bool = True,
        augment: bool = False
    ):
        """
        Initialize dataset
        
        Args:
            root_dir: Root directory containing label folders
            tokenizer: Tokenizer instance
            image_size: Target image size
            apply_preprocessing: Whether to apply manuscript preprocessing
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.apply_preprocessing = apply_preprocessing
        
        # Collect all samples
        self.samples = []
        self._collect_samples()
        
        # Define transforms
        self.base_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Optional augmentation
        if augment:
            self.augment_transform = T.Compose([
                T.RandomRotation(5),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.augment_transform = None
    
    def _collect_samples(self):
        """Collect all image paths and labels from folder structure"""
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Dataset path does not exist: {self.root_dir}")
        
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        
        for label_name in os.listdir(self.root_dir):
            label_path = os.path.join(self.root_dir, label_name)
            
            if not os.path.isdir(label_path):
                continue
            
            for img_file in os.listdir(label_path):
                _, ext = os.path.splitext(img_file.lower())
                
                if ext in valid_extensions:
                    img_path = os.path.join(label_path, img_file)
                    self.samples.append((img_path, label_name))
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {self.root_dir}")
        
        print(f"Loaded {len(self.samples)} samples from {self.root_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index
        
        Returns:
            image: Preprocessed image tensor [3, H, W]
            tokens: Encoded label tokens [seq_len]
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy sample
            image = Image.new("RGB", (self.image_size, self.image_size), color='white')
            label = "?"
        
        # Apply manuscript preprocessing if enabled
        if self.apply_preprocessing:
            image = preprocess_manuscript_image(image)
        
        # Apply augmentation if enabled
        if self.augment_transform is not None:
            image = self.augment_transform(image)
        
        # Apply standard transforms
        image = self.base_transform(image)
        
        # Tokenize label
        tokens = self.tokenizer.encode(label, add_special_tokens=True)
        
        return image, tokens


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences
    
    Args:
        batch: List of (image, tokens) tuples
        
    Returns:
        images: Stacked image tensors [batch_size, 3, H, W]
        tokens: Padded token tensors [batch_size, max_len]
    """
    images, tokens = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Pad tokens to same length
    max_len = max(len(t) for t in tokens)
    padded_tokens = []
    
    for token_seq in tokens:
        if len(token_seq) < max_len:
            padding = torch.zeros(max_len - len(token_seq), dtype=torch.long)
            padded_tokens.append(torch.cat([token_seq, padding]))
        else:
            padded_tokens.append(token_seq)
    
    tokens = torch.stack(padded_tokens)
    
    return images, tokens
