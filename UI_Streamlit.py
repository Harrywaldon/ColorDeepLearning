import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * (50 // 8) * (50 // 8), 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 9)  # 8 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply first conv layer and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply second conv layer and pooling
        x = self.pool(F.relu(self.conv3(x)))  # Apply third conv layer and pooling
        x = x.view(-1, 128 * (50 // 8) * (50 // 8))  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Apply first fully connected layer with ReLU
        x = self.dropout(x)  # Apply dropout for regularization
        x = self.fc2(x)  # Apply second fully connected layer (output layer)
        return x

## Recreate the model and load the state_dict
model = CNN()
state_dict = torch.load("color_classifier_model.pth")  # Load the state dictionary
model.load_state_dict(state_dict)         # Load into the model
model.eval()                              # Switch to evaluation mode

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((50, 50)),  # Resize to match model input size
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize if needed
])

st.title("Image Classification Interface")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

classes = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'white', 'black']

if uploaded_file is not None:
    try:
        # open image using PIL
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Make a prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction = classes[torch.argmax(output, dim=1).item()]  # Get class index

        st.write(f"Predicted Class: {prediction}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
