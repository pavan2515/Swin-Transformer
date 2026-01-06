"""
Swin Transformer Encoder
"""
import torch
import torch.nn as nn
from transformers import SwinModel


class SwinEncoder(nn.Module):
    """
    Swin Transformer encoder for image feature extraction
    """
    
    def __init__(self, model_name: str = "microsoft/swin-tiny-patch4-window7-224"):
        """
        Initialize Swin encoder
        
        Args:
            model_name: Pretrained Swin model name from HuggingFace
        """
        super().__init__()
        
        # Load pretrained Swin model
        self.swin = SwinModel.from_pretrained(model_name)
        self.hidden_dim = self.swin.config.hidden_size
        
        print(f"Loaded Swin encoder: {model_name}")
        print(f"Hidden dimension: {self.hidden_dim}")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pixel_values: Image tensor [batch_size, 3, H, W]
            
        Returns:
            Encoded features [batch_size, num_patches, hidden_dim]
        """
        outputs = self.swin(pixel_values=pixel_values)
        
        # Get last hidden state
        # Shape: [batch_size, num_patches, hidden_dim]
        return outputs.last_hidden_state
    
    def freeze_encoder(self):
        """Freeze encoder weights for fine-tuning only the decoder"""
        for param in self.swin.parameters():
            param.requires_grad = False
        print("Encoder weights frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights for full fine-tuning"""
        for param in self.swin.parameters():
            param.requires_grad = True
        print("Encoder weights unfrozen")
