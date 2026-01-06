"""
Complete Donut-style model for handwritten recognition
"""
import torch
import torch.nn as nn
from typing import Optional
import sys
sys.path.append('..')
from models.encoder import SwinEncoder
from models.decoder import TransformerDecoder


class HandwrittenDonut(nn.Module):
    """
    Donut-style model for handwritten character recognition
    
    Architecture:
    - Swin Transformer encoder for image features
    - Transformer decoder for text generation
    """
    
    def __init__(
        self,
        vocab_size: int,
        encoder_name: str = "microsoft/swin-tiny-patch4-window7-224",
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize model
        
        Args:
            vocab_size: Size of vocabulary
            encoder_name: Pretrained encoder name
            decoder_layers: Number of decoder layers
            decoder_heads: Number of attention heads in decoder
            dropout: Dropout rate
        """
        super().__init__()
        
        # Encoder
        self.encoder = SwinEncoder(encoder_name)
        
        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            hidden_dim=self.encoder.hidden_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dropout=dropout
        )
        
        self.vocab_size = vocab_size
        
        print(f"\nHandwrittenDonut Model Initialized")
        print(f"Total parameters: {self.count_parameters():,}")
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            images: Input images [batch_size, 3, H, W]
            input_ids: Target token IDs [batch_size, seq_len]
            tgt_key_padding_mask: Padding mask for targets
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        # Encode images
        encoder_outputs = self.encoder(images)
        
        # Decode
        logits = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_outputs,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return logits
    
    def generate(
        self,
        images: torch.Tensor,
        tokenizer,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 0
    ) -> torch.Tensor:
        """
        Generate text from images (inference)
        
        Args:
            images: Input images [batch_size, 3, H, W]
            tokenizer: Tokenizer instance
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = disabled)
            
        Returns:
            generated_ids: Generated token IDs [batch_size, seq_len]
        """
        self.eval()
        batch_size = images.size(0)
        device = images.device
        
        with torch.no_grad():
            # Encode images
            encoder_outputs = self.encoder(images)
            
            # Initialize with BOS token
            generated_ids = torch.full(
                (batch_size, 1),
                tokenizer.bos_token_id,
                dtype=torch.long,
                device=device
            )
            
            # Generate tokens
            for _ in range(max_length):
                # Get logits for next token
                logits = self.decoder(
                    input_ids=generated_ids,
                    encoder_hidden_states=encoder_outputs
                )
                
                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if enabled
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check if all sequences have generated EOS
                if (next_token == tokenizer.eos_token_id).all():
                    break
            
            return generated_ids
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_encoder(self):
        """Freeze encoder for fine-tuning decoder only"""
        self.encoder.freeze_encoder()
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning"""
        self.encoder.unfreeze_encoder()
