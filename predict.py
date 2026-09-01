import torch
from PIL import Image
from torchvision import transforms

from model import SatelliteCNN


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
# LOAD MODEL
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

classes = checkpoint["classes"]


model = SatelliteCNN(
    num_classes=len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)


    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    predicted_class = classes[prediction.item()]

    confidence = confidence.item() * 100


    return predicted_class, confidence


# -----------------------------
# TEST
# -----------------------------

if __name__ == "__main__":

    image_path = input(
        "Enter satellite image path: "
    )

    result, confidence = predict_image(
        image_path
    )

    print("\nPrediction:", result)
    print(
        f"Confidence: {confidence:.2f}%"
    )