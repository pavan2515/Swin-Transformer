"""
Character-level tokenizer for English and Kannada scripts
"""
import torch
from typing import List, Union


class CharTokenizer:
    """Character-level tokenizer with support for English and Kannada"""
    
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"
    
    def __init__(self, max_length: int = 10):
        """
        Initialize tokenizer with vocabulary
        
        Args:
            max_length: Maximum sequence length for padding
        """
        self.max_length = max_length
        
        # Build vocabulary
        english = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        kannada = list("ಅಆಇಈಉಊಋಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹಳ")
        numbers = list("0123456789")
        special = list(" .,;:!?'-")
        
        chars = english + kannada + numbers + special
        
        # Create vocab mappings
        self.vocab = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3
        }
        
        for i, c in enumerate(chars):
            self.vocab[c] = i + 4
            
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Special token IDs
        self.pad_token_id = self.vocab[self.PAD_TOKEN]
        self.bos_token_id = self.vocab[self.BOS_TOKEN]
        self.eos_token_id = self.vocab[self.EOS_TOKEN]
        self.unk_token_id = self.vocab[self.UNK_TOKEN]
    
    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        """
        Encode text to token IDs
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            Tensor of token IDs
        """
        ids = []
        
        if add_special_tokens:
            ids.append(self.bos_token_id)
        
        for char in text:
            ids.append(self.vocab.get(char, self.unk_token_id))
        
        if add_special_tokens:
            ids.append(self.eos_token_id)
        
        return torch.tensor(ids, dtype=torch.long)
    
    def encode_batch(self, texts: List[str], padding: bool = True) -> torch.Tensor:
        """
        Encode batch of texts with optional padding
        
        Args:
            texts: List of text strings
            padding: Whether to pad to max_length
            
        Returns:
            Tensor of shape [batch_size, seq_len]
        """
        encoded = [self.encode(text) for text in texts]
        
        if padding:
            max_len = min(max(len(e) for e in encoded), self.max_length)
            padded = []
            
            for enc in encoded:
                if len(enc) > max_len:
                    padded.append(enc[:max_len])
                else:
                    pad_len = max_len - len(enc)
                    padded.append(torch.cat([enc, torch.full((pad_len,), self.pad_token_id)]))
            
            return torch.stack(padded)
        
        return encoded
    
    def decode(self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            ids: Token IDs (list or tensor)
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        chars = []
        for token_id in ids:
            char = self.inv_vocab.get(token_id, "")
            
            if skip_special_tokens and char in [self.PAD_TOKEN, self.BOS_TOKEN, 
                                                  self.EOS_TOKEN, self.UNK_TOKEN]:
                continue
            
            chars.append(char)
        
        return "".join(chars)
    
    def decode_batch(self, batch_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode batch of token IDs
        
        Args:
            batch_ids: Tensor of shape [batch_size, seq_len]
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded strings
        """
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]
    
    def __len__(self):
        return len(self.vocab)
    
    def __repr__(self):
        return f"CharTokenizer(vocab_size={len(self)}, max_length={self.max_length})"
