import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler
import clip
from loss_functions import get_loss_function
from model_saver import ModelSaver

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a CLIP classifier on image dataset")
    parser.add_argument("--data_dir", type=str, default="engine_dataset", help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument("--loss_function", type=str, default="cce", help="Loss function to use for training")
    parser.add_argument("--precision", type=str, default="FP32", choices=['FP16', 'FP32'], help="Precision format for training")
    parser.add_argument("--save_format", type=str, default="pth", choices=['pth', 'pt', 'onnx'], help="Format to save the model")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP model to use")
    return parser.parse_args()

def setup_wandb():
    use_wandb = input("Do you want to use Weights & Biases for logging? (y/n): ").lower().strip() == 'y'
    if use_wandb:
        api_key = input("Please enter your Weights & Biases API key: ").strip()
        os.environ["WANDB_API_KEY"] = api_key
        try:
            wandb.login()
            project_name = input("Enter the Wandb project name: ").strip()
            wandb.init(project=project_name)
        except wandb.errors.AuthenticationError:
            print("Invalid API key. Weights & Biases logging will be disabled.")
            use_wandb = False
    return use_wandb

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(clip_model.visual.output_dim, num_classes)

    def forward(self, image):
        with torch.no_grad():
            features = self.clip_model.encode_image(image)
        return self.classifier(features.float())

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, use_wandb, use_amp, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = validate_model(model, criterion, val_loader, device, use_amp)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()
            best_model_path = os.path.join(save_dir, f'best_model_epoch_{epoch+1}.pth')
            torch.save(best_model_wts, best_model_path)
            print(f"New best model saved: {best_model_path}")

    model.load_state_dict(best_model_wts)
    return model

def validate_model(model, criterion, val_loader, device, use_amp):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    return val_loss, val_acc

def compute_confusion_matrix(model, data_loader, num_classes, device, use_amp):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Computing Confusion Matrix"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast(enabled=use_amp):
                outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    return cm

def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return fig

def main(args):
    save_dir = os.path.join(args.data_dir, 'best_models')
    os.makedirs(save_dir, exist_ok=True)

    use_wandb = setup_wandb()
    use_amp = args.precision == "FP16"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(args.clip_model, device=device)

    data_transforms = {
        'train': transforms.Compose([
            preprocess,
            transforms.RandomHorizontalFlip(),
        ]),
        'val': preprocess,
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    num_classes = len(image_datasets['train'].classes)
    class_names = image_datasets['train'].classes

    model = CLIPClassifier(clip_model, num_classes)

    criterion = get_loss_function(args.loss_function, num_classes)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    if use_wandb:
        wandb.watch(model)

    trained_model = train_model(model, criterion, optimizer, dataloaders['train'], dataloaders['val'], args.epochs, use_wandb, use_amp, save_dir)

    cm = compute_confusion_matrix(trained_model, dataloaders['val'], num_classes, device, use_amp)
    
    confusion_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, confusion_matrix_path)
    print(f"Confusion matrix saved as '{confusion_matrix_path}'")

    model_path = os.path.join(save_dir, f'final_model.{args.save_format}')
    input_shape = (1, 3, 224, 224)
    ModelSaver.save_model(trained_model, model_path, args.save_format, input_shape)
    
    print(f"Final model saved as '{model_path}'")

    if use_wandb:
        wandb.log({"confusion_matrix": wandb.Image(confusion_matrix_path)})
        wandb.save(model_path)
        wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)