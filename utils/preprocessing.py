"""
Image preprocessing utilities for manuscript images
"""
import cv2
import numpy as np
from PIL import Image


def preprocess_manuscript_image(image: Image.Image) -> Image.Image:
    """
    Apply manuscript-specific preprocessing to PIL image
    
    Steps:
    1. Convert to grayscale
    2. Denoise with Non-local Means
    3. Enhance contrast with CLAHE
    4. Apply adaptive thresholding
    5. Morphological operations to clean noise
    
    Args:
        image: PIL Image in RGB
        
    Returns:
        Preprocessed PIL Image in RGB
    """
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    
    # Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        contrast_enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=10
    )
    
    # Morphological operations to remove noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Convert back to RGB for consistency
    rgb_output = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    
    # Convert back to PIL
    return Image.fromarray(rgb_output)


def simple_preprocessing(image: Image.Image) -> Image.Image:
    """
    Simple preprocessing for cleaner images
    
    Args:
        image: PIL Image
        
    Returns:
        Lightly preprocessed PIL Image
    """
    # Convert to numpy
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Simple thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to RGB
    rgb_output = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(rgb_output)
