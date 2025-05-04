import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from model import EncoderCNN, DecoderRNN

def download_resnet():
    """
    Download and save the pre-trained ResNet model.
    This is a utility script to ensure the model is available
    even in offline mode.
    """
    print("Downloading pre-trained ResNet50 model...")
    # This will download the model if it's not already cached
    resnet = models.resnet50(pretrained=True)
    print("ResNet50 downloaded successfully.")

    # Create a simple model to verify it works
    print("Creating a sample encoder model...")
    encoder = EncoderCNN(256)
    print("Sample encoder created successfully.")

if __name__ == "__main__":
    download_resnet()
    print("Model preparation complete.")
