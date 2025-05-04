import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import logging
import PIL

# Try to import skimage but don't fail if it's not available
try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False

# Determine which PIL version we have and set constants accordingly
if hasattr(Image, 'Resampling'):  # PIL >= 9.1.0
    LANCZOS = Image.Resampling.LANCZOS
else:  # Older PIL versions
    LANCZOS = Image.LANCZOS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_images(image1, image2):
    """
    Compare two images and return similarity metrics.
    
    Args:
        image1: First PIL Image object
        image2: Second PIL Image object
        
    Returns:
        Dictionary containing similarity metrics
    """
    try:
        logger.info("Starting image comparison")
        # Resize both images to the same size
        # Use the smaller dimensions to ensure both can be compared
        width1, height1 = image1.size
        width2, height2 = image2.size
        
        logger.info(f"Image 1 size: {width1}x{height1}, Image 2 size: {width2}x{height2}")
        
        width = min(width1, width2)
        height = min(height1, height2)
        
        # Resize images using the appropriate LANCZOS constant for this PIL version
        img1 = image1.resize((width, height), LANCZOS)
        img2 = image2.resize((width, height), LANCZOS)
        
        # Convert to RGB if needed
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        logger.info("Images resized and converted to RGB")
        
        # Convert to numpy arrays
        array1 = np.array(img1)
        array2 = np.array(img2)
        
        # Calculate MSE (Mean Squared Error)
        try:
            mse = np.mean((array1.astype(float) - array2.astype(float)) ** 2)
            logger.info(f"MSE calculated: {mse}")
        except Exception as e:
            logger.error(f"Error calculating MSE: {str(e)}")
            mse = 0.0
        
        # Calculate SSIM (Structural Similarity Index) if scikit-image is available
        try:
            if HAVE_SKIMAGE:
                gray1 = np.array(img1.convert('L'))
                gray2 = np.array(img2.convert('L'))
                logger.info("Images converted to grayscale for SSIM")
                
                # Data range for 8-bit images is 255
                similarity_index, _ = ssim(gray1, gray2, full=True, data_range=255)
                logger.info(f"SSIM calculated: {similarity_index}")
            else:
                # Fallback to a simple similarity measure when scikit-image is not available
                logger.info("scikit-image not available, using fallback similarity measure")
                # Use normalized histogram correlation as similarity
                hist1 = img1.convert('L').histogram()
                hist2 = img2.convert('L').histogram()
                
                if sum(hist1) > 0 and sum(hist2) > 0:
                    hist1_norm = [h / sum(hist1) for h in hist1]
                    hist2_norm = [h / sum(hist2) for h in hist2]
                    similarity_index = sum(min(h1, h2) for h1, h2 in zip(hist1_norm, hist2_norm))
                else:
                    similarity_index = 0.0
                logger.info(f"Fallback similarity calculated: {similarity_index}")
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            similarity_index = 0.0
        
        # Calculate histogram similarity
        try:
            hist1 = img1.histogram()
            hist2 = img2.histogram()
            
            # Ensure histograms are non-empty
            if sum(hist1) == 0 or sum(hist2) == 0:
                hist_correlation = 0.0
            else:
                # Normalize histograms
                hist1_norm = [h / sum(hist1) for h in hist1]
                hist2_norm = [h / sum(hist2) for h in hist2]
                
                # Calculate histogram correlation
                hist_correlation = sum(np.sqrt(h1 * h2) for h1, h2 in zip(hist1_norm, hist2_norm))
            
            logger.info(f"Histogram correlation calculated: {hist_correlation}")
        except Exception as e:
            logger.error(f"Error calculating histogram correlation: {str(e)}")
            hist_correlation = 0.0
        
        # Create output dictionary
        comparison = {
            "dimensions": f"{width} x {height} pixels",
            "mse": f"{mse:.2f}",
            "ssim": f"{similarity_index:.4f}",
            "histogram_correlation": f"{hist_correlation:.4f}",
            "similarity_percentage": f"{similarity_index * 100:.2f}%"
        }
        
        logger.info("Image comparison completed successfully")
        return comparison
    except Exception as e:
        logger.error(f"Error in compare_images: {str(e)}")
        # Return a default comparison result in case of error
        return {
            "dimensions": "Unknown",
            "mse": "0.00",
            "ssim": "0.0000",
            "histogram_correlation": "0.0000",
            "similarity_percentage": "0.00%"
        }

def create_difference_image(image1, image2):
    """
    Create an image highlighting the differences between two images.
    
    Args:
        image1: First PIL Image object
        image2: Second PIL Image object
        
    Returns:
        PIL Image object showing the differences
    """
    try:
        logger.info("Creating difference image")
        # Resize both images to the same size
        width1, height1 = image1.size
        width2, height2 = image2.size
        
        width = min(width1, width2)
        height = min(height1, height2)
        
        # Resize images
        try:
            img1 = image1.resize((width, height), Image.Resampling.LANCZOS)
            img2 = image2.resize((width, height), Image.Resampling.LANCZOS)
        except AttributeError:
            # Fall back to older PIL versions
            img1 = image1.resize((width, height), Image.LANCZOS)
            img2 = image2.resize((width, height), Image.LANCZOS)
        
        # Convert to RGB if needed
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        # Convert to numpy arrays
        array1 = np.array(img1)
        array2 = np.array(img2)
        
        # Create difference image
        diff = np.abs(array1.astype(int) - array2.astype(int)).astype(np.uint8)
        
        # Convert back to PIL Image
        diff_img = Image.fromarray(diff)
        
        # Create a more visually appealing difference representation
        # Enhance the difference to make it more visible
        enhanced_diff = np.zeros_like(diff)
        # Scale up differences to make them more visible
        enhanced_diff = np.clip(diff * 2, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        enhanced_diff_img = Image.fromarray(enhanced_diff)
        
        logger.info("Difference image created successfully")
        return enhanced_diff_img
    except Exception as e:
        logger.error(f"Error creating difference image: {str(e)}")
        # Create a blank red image as fallback
        blank = Image.new('RGB', (400, 300), color=(255, 200, 200))
        draw = ImageDraw.Draw(blank)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        draw.text((100, 150), "Error creating difference image", fill=(0, 0, 0), font=font)
        return blank

def create_side_by_side_comparison(image1, image2, captions=None):
    """
    Create a side-by-side comparison image with captions.
    
    Args:
        image1: First PIL Image object
        image2: Second PIL Image object
        captions: Tuple of (caption1, caption2) or None
        
    Returns:
        PIL Image object showing both images side by side
    """
    try:
        logger.info("Creating side-by-side comparison")
        # Set default captions if not provided
        if captions is None:
            captions = ("Image 1", "Image 2")
        
        # Resize both images to the same size (using the smaller image's dimensions)
        width1, height1 = image1.size
        width2, height2 = image2.size
        
        logger.info(f"Image 1 size: {width1}x{height1}, Image 2 size: {width2}x{height2}")
        
        width = min(width1, width2)
        height = min(height1, height2)
        
        # Resize images
        try:
            img1 = image1.resize((width, height), Image.Resampling.LANCZOS)
            img2 = image2.resize((width, height), Image.Resampling.LANCZOS)
        except AttributeError:
            # Fall back to older PIL versions
            img1 = image1.resize((width, height), Image.LANCZOS)
            img2 = image2.resize((width, height), Image.LANCZOS)
        
        # Convert to RGB if needed
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        logger.info("Images resized and converted to RGB")
        
        # Calculate dimensions for the combined image
        margin = 20
        text_height = 40
        combined_width = width * 2 + margin * 3
        combined_height = height + margin * 2 + text_height
        
        # Create new image
        combined_img = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))
        
        # Paste the two images
        combined_img.paste(img1, (margin, margin))
        combined_img.paste(img2, (width + margin * 2, margin))
        
        # Add captions
        draw = ImageDraw.Draw(combined_img)
        
        # Try to get a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((margin + width // 2 - 50, height + margin + 10), captions[0], fill=(0, 0, 0), font=font)
        draw.text((width + margin * 2 + width // 2 - 50, height + margin + 10), captions[1], fill=(0, 0, 0), font=font)
        
        # Add border between images for visual separation
        for i in range(margin):
            draw.line(
                [(width + margin + i, margin), 
                 (width + margin + i, height + margin)], 
                fill=(200, 200, 200), 
                width=1
            )
        
        # Add border around the images for better visual appearance
        draw.rectangle(
            [(margin - 1, margin - 1), (margin + width + 1, margin + height + 1)],
            outline=(0, 0, 0), 
            width=1
        )
        draw.rectangle(
            [(width + margin * 2 - 1, margin - 1), (width + margin * 2 + width + 1, margin + height + 1)],
            outline=(0, 0, 0), 
            width=1
        )
        
        logger.info("Side-by-side comparison created successfully")
        return combined_img
    except Exception as e:
        logger.error(f"Error creating side-by-side comparison: {str(e)}")
        # Create a blank image as fallback
        blank = Image.new('RGB', (800, 400), color=(255, 255, 255))
        draw = ImageDraw.Draw(blank)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        draw.text((250, 180), "Error creating comparison image", fill=(0, 0, 0), font=font)
        
        # Add basic information about the original images
        try:
            draw.text((250, 220), f"Image 1: {image1.size}, {image1.mode}", fill=(0, 0, 0), font=font)
            draw.text((250, 260), f"Image 2: {image2.size}, {image2.mode}", fill=(0, 0, 0), font=font)
        except:
            pass
            
        return blank