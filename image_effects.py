import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

def apply_filter(image, filter_name):
    """
    Apply various filters and effects to an image.
    
    Args:
        image: PIL Image object
        filter_name: String name of the filter to apply
        
    Returns:
        Processed PIL Image object
    """
    # Create a copy to avoid modifying the original
    img = image.copy()
    
    if filter_name == "original":
        return img
    
    elif filter_name == "grayscale":
        return ImageOps.grayscale(img)
    
    elif filter_name == "sepia":
        # Apply sepia effect
        sepia_matrix = (
            0.393, 0.769, 0.189,
            0.349, 0.686, 0.168,
            0.272, 0.534, 0.131
        )
        sepia_img = img.convert('RGB')
        return sepia_img.convert('RGB', matrix=sepia_matrix)
    
    elif filter_name == "negative":
        return ImageOps.invert(img)
    
    elif filter_name == "blur":
        return img.filter(ImageFilter.BLUR)
    
    elif filter_name == "contour":
        return img.filter(ImageFilter.CONTOUR)
    
    elif filter_name == "emboss":
        return img.filter(ImageFilter.EMBOSS)
    
    elif filter_name == "edge_enhance":
        return img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    elif filter_name == "sharpen":
        return img.filter(ImageFilter.SHARPEN)
    
    elif filter_name == "high_contrast":
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(2.0)
    
    elif filter_name == "vivid":
        # Increase saturation and contrast for vivid effect
        color_enhancer = ImageEnhance.Color(img)
        img = color_enhancer.enhance(1.5)
        contrast_enhancer = ImageEnhance.Contrast(img)
        return contrast_enhancer.enhance(1.3)
    
    elif filter_name == "warm":
        # Add warm tone
        r, g, b = img.split()
        r = ImageEnhance.Brightness(r).enhance(1.1)
        g = ImageEnhance.Brightness(g).enhance(1.0)
        b = ImageEnhance.Brightness(b).enhance(0.9)
        return Image.merge("RGB", (r, g, b))
    
    elif filter_name == "cool":
        # Add cool tone
        r, g, b = img.split()
        r = ImageEnhance.Brightness(r).enhance(0.9)
        g = ImageEnhance.Brightness(g).enhance(1.0)
        b = ImageEnhance.Brightness(b).enhance(1.1)
        return Image.merge("RGB", (r, g, b))
    
    else:
        # Return original image if filter not found
        return img

def get_available_filters():
    """
    Get a list of all available filter names.
    
    Returns:
        List of filter names as strings
    """
    return [
        "original",
        "grayscale",
        "sepia",
        "negative",
        "blur",
        "contour",
        "emboss",
        "edge_enhance",
        "sharpen",
        "high_contrast",
        "vivid",
        "warm",
        "cool"
    ]