import torch
import clip
from PIL import Image
import os 
import shutil
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Classify images using CLIP model")
    parser.add_argument("--input", type=str, required=True, help="Path to the input folder containing images")
    parser.add_argument("--categories", nargs='+', required=True, help="List of categories for classification")
    parser.add_argument("--output_dir", type=str, default="engine_dataset", help="Name of the output folder (default: engine_dataset)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")
    return parser.parse_args()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device)

    input_folder = args.input
    output_folder = args.output
    categories = args.categories
    THRESHOLD = args.threshold

    text = clip.tokenize(categories).to(device)

    # Create main output directory
    os.makedirs(output_folder, exist_ok=True)

    # Create category subdirectories
    for category in categories:
        os.makedirs(os.path.join(output_folder, category), exist_ok=True)

    for image_name in tqdm(os.listdir(input_folder)):
        try:
            image_path = os.path.join(input_folder, image_name)
            image = transform(Image.open(image_path)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                logits_per_image, _ = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            max_prob = max(probs)
            max_prob_index = probs.tolist().index(max_prob)
            
            if max_prob >= THRESHOLD:
                category = categories[max_prob_index]
                output_path = os.path.join(output_folder, category, image_name)
                shutil.copy(image_path, output_path)
                print(f"Image {image_name} classified as {category} with probability {max_prob:.2f}")
            else:
                print(f"Image {image_name} not classified (max probability {max_prob:.2f} below threshold)")
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")

    print("Classification complete.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)