"""
Transformer Decoder for sequence generation
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for text generation
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        """
        Initialize decoder
        
        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension size
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        print(f"Initialized Transformer Decoder:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Layers: {num_layers}")
        print(f"  Heads: {num_heads}")
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Target token IDs [batch_size, tgt_seq_len]
            encoder_hidden_states: Encoder outputs [batch_size, src_seq_len, hidden_dim]
            tgt_mask: Target attention mask
            tgt_key_padding_mask: Target padding mask
            
        Returns:
            logits: Output logits [batch_size, tgt_seq_len, vocab_size]
        """
        # Embed tokens
        embedded = self.embedding(input_ids) * math.sqrt(self.hidden_dim)
        
        # Apply positional encoding
        # Note: pos_encoder expects [seq_len, batch_size, hidden_dim]
        embedded = embedded.transpose(0, 1)
        embedded = self.pos_encoder(embedded)
        embedded = embedded.transpose(0, 1)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(input_ids.size(1)).to(input_ids.device)
        
        # Decode
        decoded = self.transformer_decoder(
            tgt=embedded,
            memory=encoder_hidden_states,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoded)
        
        return logits
    
    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
