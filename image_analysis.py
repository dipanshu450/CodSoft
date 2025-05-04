import numpy as np
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

def analyze_image(image):
    """
    Analyze an image and return statistics.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary containing image statistics and analysis
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Basic image information
    width, height = image.size
    num_pixels = width * height
    
    # Get image statistics
    stat = ImageStat.Stat(image)
    
    # Calculate color distribution
    r, g, b = image.split()
    r_values = np.array(r).flatten()
    g_values = np.array(g).flatten()
    b_values = np.array(b).flatten()
    
    # Calculate brightness
    brightness = sum(stat.mean) / 3
    
    # Calculate contrast
    contrast = sum(stat.stddev) / 3
    
    # Calculate dominant color
    # Simplify to a 5x5x5 color cube
    r_simplified = (r_values // 51) * 51
    g_simplified = (g_values // 51) * 51
    b_simplified = (b_values // 51) * 51
    
    # Combine into color tuples
    colors = np.column_stack((r_simplified, g_simplified, b_simplified))
    
    # Find the most common color
    unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
    dominant_idx = np.argmax(counts)
    dominant_color = unique_colors[dominant_idx]
    
    # Convert dominant color to hex for display
    dominant_color_hex = '#{:02x}{:02x}{:02x}'.format(
        int(dominant_color[0]), 
        int(dominant_color[1]), 
        int(dominant_color[2])
    )
    
    # Determine if image is mostly dark or light
    if brightness < 128:
        tonality = "Dark"
    else:
        tonality = "Light"
    
    # Determine if image is high or low contrast
    if contrast < 50:
        contrast_desc = "Low"
    else:
        contrast_desc = "High"
    
    # Create output dictionary
    analysis = {
        "dimensions": f"{width} x {height} pixels",
        "file_size": f"{num_pixels * 3 / 1024 / 1024:.2f} MB (estimated)",
        "brightness": f"{brightness:.1f} / 255 ({tonality})",
        "contrast": f"{contrast:.1f} / 255 ({contrast_desc})",
        "color_balance": {
            "red": f"{stat.mean[0]:.1f}",
            "green": f"{stat.mean[1]:.1f}",
            "blue": f"{stat.mean[2]:.1f}"
        },
        "dominant_color": dominant_color_hex,
        "format": image.format or "Unknown",
        "mode": image.mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return analysis

def generate_color_histogram(image):
    """
    Generate color histograms for an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded PNG image of the histograms
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Split channels
    r, g, b = image.split()
    
    # Create histograms
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    axs[0].hist(np.array(r).flatten(), bins=256, range=(0, 256), color='red', alpha=0.7)
    axs[0].set_title('Red Channel')
    axs[0].set_xlim(0, 256)
    axs[0].set_ylim(bottom=0)
    
    axs[1].hist(np.array(g).flatten(), bins=256, range=(0, 256), color='green', alpha=0.7)
    axs[1].set_title('Green Channel')
    axs[1].set_xlim(0, 256)
    axs[1].set_ylim(bottom=0)
    
    axs[2].hist(np.array(b).flatten(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    axs[2].set_title('Blue Channel')
    axs[2].set_xlim(0, 256)
    axs[2].set_ylim(bottom=0)
    
    fig.suptitle('Color Histograms')
    plt.tight_layout()
    
    # Save figure to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode as base64
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return data

def estimate_image_quality(image):
    """
    Estimate image quality based on various metrics.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary containing quality metrics
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Basic image information
    width, height = image.size
    
    # Calculate metrics
    
    # Resolution quality (based on total pixels)
    pixels = width * height
    if pixels >= 4000000:  # 4 megapixels or higher
        resolution_quality = "Excellent"
        resolution_score = 5
    elif pixels >= 2000000:  # 2 megapixels or higher
        resolution_quality = "Very Good"
        resolution_score = 4
    elif pixels >= 1000000:  # 1 megapixel or higher
        resolution_quality = "Good"
        resolution_score = 3
    elif pixels >= 500000:  # 0.5 megapixels or higher
        resolution_quality = "Fair"
        resolution_score = 2
    else:
        resolution_quality = "Poor"
        resolution_score = 1
    
    # Get image statistics for noise and contrast estimation
    stat = ImageStat.Stat(image)
    
    # Contrast quality (based on standard deviation of pixel values)
    contrast = sum(stat.stddev) / 3
    if contrast >= 70:
        contrast_quality = "Excellent"
        contrast_score = 5
    elif contrast >= 50:
        contrast_quality = "Very Good"
        contrast_score = 4
    elif contrast >= 30:
        contrast_quality = "Good"
        contrast_score = 3
    elif contrast >= 15:
        contrast_quality = "Fair"
        contrast_score = 2
    else:
        contrast_quality = "Poor"
        contrast_score = 1
    
    # Estimate overall quality (simple average)
    overall_score = (resolution_score + contrast_score) / 2
    
    if overall_score >= 4.5:
        overall_quality = "Excellent"
    elif overall_score >= 3.5:
        overall_quality = "Very Good"
    elif overall_score >= 2.5:
        overall_quality = "Good"
    elif overall_score >= 1.5:
        overall_quality = "Fair"
    else:
        overall_quality = "Poor"
    
    # Create output dictionary
    quality = {
        "resolution_quality": resolution_quality,
        "contrast_quality": contrast_quality,
        "overall_quality": overall_quality,
        "overall_score": f"{overall_score:.1f}/5"
    }
    
    return quality