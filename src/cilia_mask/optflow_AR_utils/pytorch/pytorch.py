import numpy as np
import os.path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset, random_split


    


# Define the FCN model with ResNet-50 backbone
class FCNResNet(nn.Module):
    def __init__(self, num_classes):
        super(FCNResNet, self).__init__()
        self.resnet = models.segmentation.fcn_resnet50()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size = 7)
        self.resnet.fc = nn.Conv2d(2048, num_classes, kernel_size = 1)
    
    def forward(self, x):
        x = self.resnet(x)
        x = torch.nn.functional.interpolate(x, size = (480, 640), mode='bilinear', align_corners=False)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set parameters
num_classes = 1
batch_size = 16
lr = 0.001
num_epochs = 10

# Define transforms
transform = transforms.Compose([
    transforms.ToImageTensor(),
    transforms.Grayscale(1),
    transforms.Resize(size = (480, 640), interpolation = transforms.InterpolationMode.NEAREST_EXACT),
    transforms.ConvertDtype(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Load your custom dataset
train_dataset = CiliaImages(root='/space/cilia/optflow', flow_type = "nvidia", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
i, m = train_dataset[0]
np.save("mask.npy", m)
print(train_dataset.data_dirs[0])
quit()
# Create an instance of FCNResNet
model = FCNResNet(num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
total_steps = len(train_dataloader)
print(total_steps)
for i, (images, masks) in enumerate(train_dataloader):
    image = images[0]
    mask = masks[0]
    m = mask.detach().cpu().numpy()
    img = image.detach().cpu().numpy()
    np.save(f"{i}_img.npy", img)
    np.save(f"{i}_msk.npy", m)
    print(image.size())
    print(mask.size())
quit()

for epoch in range(num_epochs):
    for i, (images, masks) in enumerate(train_dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print training progress
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}")