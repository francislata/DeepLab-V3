import torchvision.transforms as transforms
import torch
from PIL import Image
from resnet import resnet101


if __name__ == "__main__":
    # Note: This is only used for testing purposes. This will be removed at some point.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet101(pretrained=False).to(device)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open("dog.jpg")
    img_tensor = img_transforms(img).unsqueeze(0).to(device)
    _ = model(img_tensor)
