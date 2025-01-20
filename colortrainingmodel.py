import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set the base directory for the dataset
base_dir = './color_images/'
classes = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'white', 'grey' 'black']
# Parameters
img_height, img_width = 50, 50
batch_size = 16
epochs = 10  # Set number of epochs
learning_rate = 0.0001
random_seed = 32

# Set the random seed for reproducibility
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(img_height, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = datasets.ImageFolder(base_dir, transform=transform)

# Split the dataset filenames into training (60%), validation (20%), and testing (20%)
filenames = np.array(dataset.samples)[:, 0]
train_files, test_files = train_test_split(filenames, test_size=0.4, random_state=random_seed)
val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=random_seed)


# Function to get indices for files
def get_indices(files, dataset):
    indices = []
    for idx, (path, _) in enumerate(dataset.samples):
        if path in files:
            indices.append(idx)
    return indices


# Get indices for training, validation, and test datasets
train_indices = get_indices(train_files, dataset)
val_indices = get_indices(val_files, dataset)
test_indices = get_indices(test_files, dataset)

# Create subset datasets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * (img_height // 8) * (img_width // 8), 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 9)  # Creates output layer, predicting among 9 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply first conv layer and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply second conv layer and pooling
        x = self.pool(F.relu(self.conv3(x)))  # Apply third conv layer and pooling
        x = x.view(-1, 128 * (img_height // 8) * (img_width // 8))  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Apply first fully connected layer with ReLU
        x = self.dropout(x)  # Apply dropout for regularization
        x = self.fc2(x)  # Apply second fully connected layer (output layer)
        return x


model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store loss values for plotting
train_losses = []
val_losses = []

# Training the model
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = correct / total

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), 'color_classifier_model.pth')

# Evaluate the model on test set and print accuracy for each class
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
print(f"Test accuracy: {correct / total}")

# Print classification report
class_names = dataset.classes
print(classification_report(all_labels, all_preds, target_names=class_names))

# Create confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("color_confusion_matrix")
