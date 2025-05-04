import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def get_device():
    """Get the device to use (GPU if available, otherwise CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_image(image):
    """Preprocess an image for the model."""
    try:
        # Define the image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Make sure image is in correct format
        if not isinstance(image, Image.Image):
            # Try to convert from numpy array if needed
            try:
                image = Image.fromarray(np.uint8(image))
            except:
                print("Error: Image format not recognized")
                return None
        
        # Check image mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply the transformations
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        # Return None if preprocessing fails
        return None

def generate_random_caption():
    """Generate a random caption for demonstration purposes."""
    subjects = ['A person', 'A dog', 'A cat', 'A child', 'A woman', 'A man']
    verbs = ['is sitting', 'is standing', 'is walking', 'is running', 'is playing']
    locations = ['in a park', 'on the beach', 'in a room', 'at home', 'on a street']
    extras = ['with a friend', 'under a tree', 'near the water', 'on a sunny day', '']
    
    subject = np.random.choice(subjects)
    verb = np.random.choice(verbs)
    location = np.random.choice(locations)
    extra = np.random.choice(extras)
    
    caption = f"{subject} {verb} {location}"
    if extra:
        caption += f" {extra}"
    caption += "."
    
    return caption
