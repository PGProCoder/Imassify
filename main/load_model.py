from torch import nn
import torch.nn.functional as F

class ImgClfNet(nn.Module):
    def __init__(self):
        super(ImgClfNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = ImgClfNet()

PATH = './model_final.pth'
from PIL import Image
import torchvision.transforms as transforms

# Load the image
image = Image.open('.jpg') #give image path. ONLY JPG!

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to 32x32 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the tensor
])

# Apply the transformation
preprocessed_image = transform(image)

# Add an extra dimension for batch size
image = preprocessed_image.unsqueeze(0)
import torch
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net.load_state_dict(torch.load(PATH))

# Assuming that `image` is your preprocessed image
outputs = net(image)
_, predicted = torch.max(outputs.data, 1)

print("it's a", classes[predicted[0]])
