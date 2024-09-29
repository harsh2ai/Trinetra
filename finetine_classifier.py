import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a classifier on CLIP-classified dataset")
    parser.add_argument("--data_dir", type=str, default="engine_dataset", help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading")
    return parser.parse_args()

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue

        images = [img for img in os.listdir(category_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        num_images = len(images)
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)
        num_test = num_images - num_train - num_val

        train_imgs, val_imgs, test_imgs = random_split(images, [num_train, num_val, num_test])

        for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            split_dir = os.path.join(data_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)
            for img in imgs:
                src = os.path.join(category_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy(src, dst)

    for category in os.listdir(data_dir):
        if category not in ['train', 'val', 'test']:
            shutil.rmtree(os.path.join(data_dir, category))

def prepare_data(data_dir):
    if all(split in os.listdir(data_dir) for split in ['train', 'val', 'test']):
        print("Train, val, and test splits already present.")
    else:
        print("Automatically splitting data into train, val, and test sets.")
        split_dataset(data_dir)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model

def main(args):
    prepare_data(args.data_dir)

    # Data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the dataset
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    # Get the number of classes
    num_classes = len(image_datasets['train'].classes)

    # Load a pretrained ResNet50 model and modify the final layer
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    trained_model = train_model(model, criterion, optimizer, dataloaders['train'], dataloaders['val'], args.epochs)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'finetuned_model.pth')
    print("Model saved as 'finetuned_model.pth'")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)