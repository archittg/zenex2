import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SatelliteCNN


# -----------------------------
# SETTINGS
# -----------------------------

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_PATH = "satellite_model.pth"


# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# LOAD DATASET
# -----------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("Classes:", train_dataset.classes)


# -----------------------------
# CREATE MODEL
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


model = SatelliteCNN(
    num_classes=len(train_dataset.classes)
)

model = model.to(device)


# -----------------------------
# LOSS + OPTIMIZER
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# TRAINING
# -----------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Prediction
        outputs = model(images)

        # Calculate error
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )


# -----------------------------
# SAVE MODEL
# -----------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": train_dataset.classes
    },
    MODEL_PATH
)

print("\nModel saved as:", MODEL_PATH)