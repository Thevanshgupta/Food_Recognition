import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load pre-trained ResNet50 model (Food-101 fine-tuned)
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 101)  # 101 classes for Food-101
model.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://github.com/jinglescode/pytorch-pretrained-food101/releases/download/v1.0/food101_resnet50.pth",
    map_location=torch.device('cpu')
))
model.eval()

# Define Food-101 class names (shortened here for demo, full list needed)
FOOD_CLASSES = [
    "apple pie", "baby back ribs", "baklava", "beef carpaccio", "beef tartare",
    "beet salad", "beignets", "bibimbap", "bread pudding", "breakfast burrito",
    # ... (add all 101 classes here)
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_food(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
        food_name = FOOD_CLASSES[predicted.item()]
    return food_name
