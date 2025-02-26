# Concept-Based Image Generation with Stable Diffusion

This project implements a custom image generation system using Stable Diffusion with Textual Inversion. It allows you to train the model on custom concepts and generate images using these learned concepts through a user-friendly Streamlit interface.

## Features

- Train Stable Diffusion on custom concepts using Textual Inversion
- Support for multiple concepts (Airplane, Samurai, Hulk, Lego, Dolphin)
- Interactive Streamlit interface for image generation
- Customizable generation parameters
- Image download functionality
- Real-time image preview

## Requirements

- Python 3.7+
- PyTorch
- CUDA-capable GPU (recommended)

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Custom Concepts

1. Prepare your training images:
   - Create directories for each concept in `training_images/` (e.g., `airplane`, `warrior`, etc.)
   - Add 5 training images to respective directories

2. Run the training script:
   ```bash
   python train_textual_inversion.py
   ```

   The script will:
   - Train the model on each concept (airplane, warrior, monster, toy, dolphin)
   - Save learned embeddings in respective directories (embeddings_<concept>)
   - Show training progress with tqdm

### Generating Images

1. Start the Streamlit interface:
   ```bash
   streamlit run generate_images.py
   ```

2. Using the interface:
   - Select a concept from the dropdown menu (Plane, Samurai, Hulk, Lego, Dolphin)
   - Enter a prompt describing the image you want to generate
   - Adjust generation parameters:
     - Number of images (1-4)
     - Guidance Scale (1.0-20.0)
     - Number of Inference Steps (1-100)
   - Click "Generate" to create images
   - Download generated images using the provided buttons

## Model Architecture

- Base Model: Stable Diffusion 2.0
- Training Method: Textual Inversion
- Scheduler: DPMSolverMultistepScheduler

## Parameters

### Training Parameters
- Learning Rate: 5e-4
- Batch Size: 1
- Training Steps: 100 epochs
- Image Size: 512x512
- Mixed Precision: fp16
- Data Augmentation: Random crop, horizontal flip (p=0.5)

### Generation Parameters
- Guidance Scale: 1.0-20.0 (default: 7.5)
- Inference Steps: 1-100 (default: 50)
- Number of Images: 1-4

## Project Structure

```
├── train_textual_inversion.py  # Training script
├── generate_images.py          # Streamlit interface
├── requirements.txt            # Dependencies
├── training_images/            # Training data for concepts
│   ├── airplane/              # Airplane concept images
│   ├── warrior/               # Warrior concept images
│   ├── monster/               # Monster concept images
│   ├── toy/                   # Toy concept images
│   └── dolphin/               # Dolphin concept images
└── embeddings_<concept>/      # Learned embeddings
```

## Limitations

- Requires significant GPU memory for training
- Training time increases with the number of concepts
- Image generation time depends on the selected parameters

## Tips

1. For best results:
   - Use clear, high-quality training images
   - Ensure training images are diverse within each concept
   - Use descriptive prompts when generating images

2. Optimization:
   - Adjust guidance scale for better image quality
   - Increase inference steps for more detailed results
   - Use appropriate number of training epochs