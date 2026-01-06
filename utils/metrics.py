"""
Evaluation metrics for handwritten recognition
"""
import torch
from typing import List


def compute_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute exact match accuracy
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        Accuracy as percentage
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    correct = sum(pred == target for pred, target in zip(predictions, targets))
    return 100.0 * correct / len(predictions) if len(predictions) > 0 else 0.0


def compute_character_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute character-level accuracy
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        Character accuracy as percentage
    """
    total_chars = 0
    correct_chars = 0
    
    for pred, target in zip(predictions, targets):
        # Pad to same length
        max_len = max(len(pred), len(target))
        pred_padded = pred.ljust(max_len)
        target_padded = target.ljust(max_len)
        
        for p_char, t_char in zip(pred_padded, target_padded):
            total_chars += 1
            if p_char == t_char:
                correct_chars += 1
    
    if total_chars == 0:
        return 0.0
    
    return 100.0 * correct_chars / total_chars


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein distance between two strings
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Compute Character Error Rate (CER)
    
    CER = (Substitutions + Insertions + Deletions) / Total Characters
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        CER as percentage
    """
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(predictions, targets):
        distance = levenshtein_distance(pred, target)
        total_distance += distance
        total_length += len(target)
    
    if total_length == 0:
        return 0.0
    
    return 100.0 * total_distance / total_length


class MetricsTracker:
    """Track and compute metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.total_loss = 0.0
        self.num_batches = 0
    
    def update(self, preds: List[str], targets: List[str], loss: float = None):
        """
        Update metrics with new batch
        
        Args:
            preds: Predicted strings
            targets: Target strings
            loss: Optional loss value
        """
        self.predictions.extend(preds)
        self.targets.extend(targets)
        
        if loss is not None:
            self.total_loss += loss
            self.num_batches += 1
    
    def compute(self) -> dict:
        """
        Compute all metrics
        
        Returns:
            Dictionary of metric names and values
        """
        if len(self.predictions) == 0:
            return {}
        
        metrics = {
            'accuracy': compute_accuracy(self.predictions, self.targets),
            'char_accuracy': compute_character_accuracy(self.predictions, self.targets),
            'cer': compute_cer(self.predictions, self.targets),
        }
        
        if self.num_batches > 0:
            metrics['loss'] = self.total_loss / self.num_batches
        
        return metrics
    
    def __repr__(self):
        metrics = self.compute()
        return ' | '.join([f"{k}: {v:.2f}" for k, v in metrics.items()])
