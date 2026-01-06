"""
Inference script for Handwritten Character Recognition
"""
import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

from data.tokenizer import CharTokenizer
from models.model import HandwrittenDonut
from utils.preprocessing import preprocess_manuscript_image
from utils.training import load_checkpoint, load_config


class HandwritingRecognizer:
    """Inference wrapper for handwriting recognition"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize recognizer
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # Initialize tokenizer
        self.tokenizer = CharTokenizer(max_length=config['tokenizer']['max_length'])
        
        # Initialize model
        self.model = HandwrittenDonut(
            vocab_size=len(self.tokenizer),
            encoder_name=config['model']['encoder'],
            decoder_layers=config['model']['decoder_layers'],
            decoder_heads=config['model']['decoder_heads']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Store config
        self.config = config
        self.apply_preprocessing = config['data']['apply_manuscript_preprocessing']
        
        # Define transforms
        self.transform = T.Compose([
            T.Resize((config['data']['image_size'], config['data']['image_size'])),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Vocabulary size: {len(self.tokenizer)}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply manuscript preprocessing if enabled
        if self.apply_preprocessing:
            image = preprocess_manuscript_image(image)
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)
        
        return image_tensor
    
    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 0
    ) -> str:
        """
        Predict text from image
        
        Args:
            image_path: Path to image file
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            
        Returns:
            Predicted text string
        """
        # Preprocess image
        image = self.preprocess_image(image_path).to(self.device)
        
        # Generate
        generated_ids = self.model.generate(
            image,
            self.tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k
        )
        
        # Decode
        prediction = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return prediction
    
    def predict_batch(
        self,
        image_paths: list,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 0
    ) -> list:
        """
        Predict text from multiple images
        
        Args:
            image_paths: List of image paths
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            List of predicted text strings
        """
        predictions = []
        
        for img_path in image_paths:
            pred = self.predict(img_path, max_length, temperature, top_k)
            predictions.append(pred)
        
        return predictions


def main(args):
    """Main inference function"""
    
    # Initialize recognizer
    recognizer = HandwritingRecognizer(
        checkpoint_path=args.checkpoint,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Single image inference
    if args.image:
        print(f"\nProcessing image: {args.image}")
        prediction = recognizer.predict(
            args.image,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(f"Prediction: {prediction}")
    
    # Batch inference
    elif args.image_dir:
        print(f"\nProcessing images in: {args.image_dir}")
        
        # Get all images
        image_paths = []
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        
        for ext in valid_extensions:
            image_paths.extend(Path(args.image_dir).glob(f'*{ext}'))
        
        print(f"Found {len(image_paths)} images")
        
        # Predict
        predictions = recognizer.predict_batch(
            [str(p) for p in image_paths],
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        # Print results
        print("\nResults:")
        for img_path, pred in zip(image_paths, predictions):
            print(f"{img_path.name}: {pred}")
        
        # Save results if output path specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for img_path, pred in zip(image_paths, predictions):
                    f.write(f"{img_path.name}\t{pred}\n")
            print(f"\nResults saved to: {args.output}")
    
    else:
        print("Please specify either --image or --image_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for Handwritten Recognition')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to directory of images')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results')
    parser.add_argument('--max_length', type=int, default=20,
                        help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Top-k sampling (0 = disabled)')
    
    args = parser.parse_args()
    main(args)
