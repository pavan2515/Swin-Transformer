
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.tokenizer import CharTokenizer
from data.dataset import HandwrittenDataset, collate_fn
from models.model import HandwrittenDonut
from utils.metrics import MetricsTracker
from utils.training import (
    setup_logging, set_seed, load_config, save_config,
    save_checkpoint, load_checkpoint, EarlyStopping, get_lr
)


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, tokenizer, epoch, logger):
    """Train for one epoch"""
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, tokens) in enumerate(pbar):
        images = images.to(device)
        tokens = tokens.to(device)
        
        # Prepare inputs and targets
        input_ids = tokens[:, :-1]  # All except last token
        target_ids = tokens[:, 1:]  # All except first token
        
        # Create padding mask
        padding_mask = (input_ids == tokenizer.pad_token_id)
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        with autocast(enabled=(scaler is not None)):
            outputs = model(images, input_ids, tgt_key_padding_mask=padding_mask)
            
            # Compute loss
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                target_ids.reshape(-1)
            )
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Decode predictions for metrics
        with torch.no_grad():
            pred_ids = outputs.argmax(dim=-1)
            predictions = tokenizer.decode_batch(pred_ids)
            targets = tokenizer.decode_batch(target_ids)
        
        # Update metrics
        metrics.update(predictions, targets, loss.item())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute epoch metrics
    epoch_metrics = metrics.compute()
    logger.info(f"Train Epoch {epoch}: {metrics}")
    
    return epoch_metrics


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, tokenizer, epoch, logger):
    """Validate for one epoch"""
    model.eval()
    metrics = MetricsTracker()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    for images, tokens in pbar:
        images = images.to(device)
        tokens = tokens.to(device)
        
        # Prepare inputs and targets
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]
        
        # Create padding mask
        padding_mask = (input_ids == tokenizer.pad_token_id)
        
        # Forward pass
        outputs = model(images, input_ids, tgt_key_padding_mask=padding_mask)
        
        # Compute loss
        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            target_ids.reshape(-1)
        )
        
        # Decode predictions
        pred_ids = outputs.argmax(dim=-1)
        predictions = tokenizer.decode_batch(pred_ids)
        targets = tokenizer.decode_batch(target_ids)
        
        # Update metrics
        metrics.update(predictions, targets, loss.item())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute epoch metrics
    epoch_metrics = metrics.compute()
    logger.info(f"Val Epoch {epoch}: {metrics}")
    
    return epoch_metrics


def main(args):
    """Main training function"""
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup
    set_seed(config['experiment']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config['paths']['log_dir'], config['experiment']['name'])
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"Device: {device}")
    
    # Save config
    save_config(config, os.path.join(config['paths']['checkpoint_dir'], 'config.yaml'))
    
    # Initialize tokenizer
    tokenizer = CharTokenizer(max_length=config['tokenizer']['max_length'])
    logger.info(f"Tokenizer: {tokenizer}")
    
    # Load dataset
    logger.info("Loading dataset...")
    full_dataset = HandwrittenDataset(
        root_dir=config['data']['train_path'],
        tokenizer=tokenizer,
        image_size=config['data']['image_size'],
        apply_preprocessing=config['data']['apply_manuscript_preprocessing'],
        augment=True
    )
    
    # Split dataset
    val_size = int(len(full_dataset) * config['data']['val_split'])
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['experiment']['seed'])
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = HandwrittenDonut(
        vocab_size=len(tokenizer),
        encoder_name=config['model']['encoder'],
        decoder_layers=config['model']['decoder_layers'],
        decoder_heads=config['model']['decoder_heads']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # Mixed precision training
    scaler = GradScaler() if config['training']['mixed_precision'] else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        mode='min'
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config['experiment']['resume_from']:
        logger.info(f"Resuming from checkpoint: {config['experiment']['resume_from']}")
        checkpoint = load_checkpoint(
            config['experiment']['resume_from'],
            model,
            optimizer,
            device
        )
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        logger.info(f"Learning rate: {get_lr(optimizer):.6f}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, tokenizer, epoch+1, logger
        )
        
        # Validate
        if (epoch + 1) % config['training']['validate_every'] == 0:
            val_metrics = validate_epoch(
                model, val_loader, criterion, device, tokenizer, epoch+1, logger
            )
            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                logger.info(f"New best model! Val Loss: {best_val_loss:.4f}")
            
            if (epoch + 1) % config['training']['save_every'] == 0 or is_best:
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'best_val_loss': best_val_loss,
                        'config': config
                    },
                    config['paths']['checkpoint_dir'],
                    f'checkpoint_epoch_{epoch+1}.pth',
                    is_best
                )
            
            # Early stopping
            if early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Step scheduler
        scheduler.step()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Handwritten Recognition Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    main(args)
