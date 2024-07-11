import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

label_set = ['bordered','borderless','row_bordered']
id2label = {v: k for v, k in enumerate(label_set)}
label2id = {k: v for v, k in enumerate(label_set)}

def load_model():

    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3)
    )

    model = model.to(device)
    # make sure to place the resnet.pth file(trained model) in the same directory of the code file.
    checkpoint = torch.load('./resnet.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def predict(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

    predicted_class, probs=  predicted.item(), probabilities
    return label_set[predicted_class], probs[0][predicted_class].item()

model = load_model()

# give image path here
# example
image_path = r'test/result_table_img_86070_0_1707342385.jpg'

print(predict(image_path))


