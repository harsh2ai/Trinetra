# 👁️ Trinetra: Advanced Image Classification Pipeline

Trinetra, named after the "third eye" in Hindu mythology, is a comprehensive image classification pipeline that streamlines the process from data labelling to model training and evaluation. It provides clear vision into your image data using state-of-the-art CLIP models.

## 🌟 Features

- 🏷️ Automated image labelling for efficient dataset preparation
- 🧠 Fine-tuning of CLIP models for custom image classification tasks
- 🔢 Support for multiple precision formats (FP16, FP32)
- 📊 Integration with Weights & Biases for experiment tracking
- 💾 Automatic saving of best-performing models
- 🧮 Confusion matrix computation for performance analysis
- 📁 Flexible model saving in different formats (PTH, PT, ONNX)

## 🛠️ Installation

### Automated Setup

1. Ensure you have Anaconda or Miniconda installed.
2. Clone the repository:
   ```
   git clone https://github.com/your-username/Trinetra.git
   cd Trinetra
   ```
3. Run the setup script generator:
   ```
   python generate_setup.py
   ```
4. Follow the instructions to run the generated setup script for your operating system.

### Manual Setup

If you prefer to set up manually:

1. Create and activate a new Conda environment:
   ```
   conda create --name trinetra python=3.8
   conda activate trinetra
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   pip install git+https://github.com/openai/CLIP.git
   ```

## 🚀 Usage

### Image Labelling

To label your images using CLIP:

```
python src/classifier_labelling.py --data_dir "path/to/unlabelled/images" --output_dir "path/to/labelled/output"
```

### Model Training

To fine-tune a CLIP model on your labelled dataset:

```
python src/finetine_classifier.py --data_dir "path/to/labelled/dataset" --epochs 10 --batch_size 32 --learning_rate 0.001 --loss_function focal --precision FP16 --save_format pt
```

### Command-line Arguments

- `--data_dir`: Path to the dataset directory
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for the optimizer
- `--loss_function`: Loss function to use (e.g., "focal", "cce")
- `--precision`: Precision format for training ("FP16" or "FP32")
- `--save_format`: Format to save the model ("pth", "pt", or "onnx")
- `--clip_model`: CLIP model to use (default: "ViT-B/32")

For a full list of options, run:
```
python src/finetine_classifier.py --help
```

## 📁 Project Structure

```
Trinetra/
│
├── src/
│   ├── classifier_labelling.py
│   ├── finetine_classifier.py
│   ├── loss_functions.py
│   ├── precision_formats.py
│   └── model_saver.py
│
├── requirements.txt
├── generate_setup.py
├── README.md
└── .gitignore
```

## 🔮 Upcoming Features

- Object Detection capabilities
- Semantic and instance segmentation support
- Advanced data augmentation techniques
- Multi-GPU and distributed training optimization
- Model interpretability tools

## 🤝 Contributing

Contributions to Trinetra are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository on GitHub.
2. Clone your forked repository locally.
3. Create a new branch for your feature or bug fix.
4. Make your changes and commit them with clear, descriptive messages.
5. Push your changes to your fork on GitHub.
6. Submit a Pull Request to the main Trinetra repository.

## 📊 Performance

Trinetra has been tested on various datasets and consistently achieves high accuracy in image classification tasks. Specific performance metrics will vary based on the dataset and chosen model architecture.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📞 Contact

For any queries or suggestions, please open an issue on GitHub or contact the maintainer at [your-email@example.com].

---

Empower Your Vision with Trinetra! 👁️🚀

```
         _____
     _.-'     `'-._
   ,'               '.
  /                   \
 |    ___________     |
 |   |           |    |
 |   |    ( )    |    |
 |   |___________| 👁️  |
  \                   /
   '.               ,'
     '-.._______.-'

 Trinetra watches everything
```