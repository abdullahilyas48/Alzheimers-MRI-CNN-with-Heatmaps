# Interpretable CNN Classification of Alzheimer's Disease via MRI Heatmaps

This project implements an interpretable Convolutional Neural Network (CNN) for classifying Alzheimer's disease stages using MRI brain scans. The model uses ResNet50 architecture with Grad-CAM visualization to provide explainable predictions through heatmaps that highlight regions of the brain most influential to the classification decision.

## 🎯 Project Overview

Alzheimer's disease is a progressive neurodegenerative disorder that affects millions worldwide. Early detection and accurate staging are crucial for effective treatment and patient care. This project leverages deep learning to classify MRI images into four Alzheimer's disease stages:

- **Non Demented**: Normal cognitive function
- **Very Mild Demented**: Early stage with minimal symptoms
- **Mild Demented**: Noticeable cognitive decline
- **Moderate Demented**: Significant impairment requiring assistance

## 🏗️ Architecture & Methodology

### Model Architecture
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Input**: Grayscale MRI images (224×224 pixels)
- **Preprocessing**: Channel expansion (1→3 channels) for ResNet compatibility
- **Classification Head**: Global Average Pooling → Dropout (0.5) → Dense (4 classes, softmax)

### Training Strategy
The model employs a two-phase training approach:

1. **Phase 1**: Frozen backbone training
   - ResNet50 layers frozen
   - Learning rate: 1e-4
   - Early stopping with patience=5

2. **Phase 2**: Fine-tuning
   - All layers trainable
   - Learning rate: 1e-5
   - Extended training with early stopping

### Data Augmentation
- Rotation (±15°)
- Width/height shift (10%)
- Horizontal/vertical flipping
- Fill mode: nearest

## 📊 Performance Results

The model achieves **98.6% validation accuracy** with the following metrics:

| Stage | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Mild Demented | 0.99 | 0.98 | 0.98 | 179 |
| Moderate Demented | 1.00 | 1.00 | 1.00 | 12 |
| Non Demented | 0.98 | 0.99 | 0.98 | 640 |
| Very Mild Demented | 0.98 | 0.97 | 0.97 | 448 |

**Overall Accuracy**: 98.6%

## 🔍 Interpretability with Grad-CAM

The project implements Gradient-weighted Class Activation Mapping (Grad-CAM) to provide visual explanations:

- **Heatmap Generation**: Identifies brain regions most relevant to predictions
- **Stability Analysis**: SSIM-based evaluation of heatmap consistency under noise
- **Visualization**: Superimposed heatmaps on original MRI scans

Key features:
- Automatic detection of last convolutional layer
- Channel-wise gradient weighting
- Color-coded heatmap overlay
- Batch processing for multiple images

## 📁 Project Structure

```
Project/
├── alz_split/                    # Train/val/test splits
│   ├── train/
│   ├── val/
│   └── test/
├── combined_images/              # Raw dataset
├── DL_project_Resnet.ipynb       # Main implementation
├── alzheimer_resnet50_model.keras # Final trained model
├── phase1_model.keras           # Phase 1 model checkpoint
├── phase2_model.keras           # Phase 2 model checkpoint
├── phase1_log.csv              # Phase 1 training logs
└── phase2_log.csv              # Phase 2 training logs
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow-macos tensorflow-metal  # For Apple Silicon
pip install torch torchvision                 # PyTorch with MPS support
pip install split-folders pandas scikit-learn seaborn matplotlib opencv-python
```

### Dataset Setup

1. Download the Alzheimer's MRI dataset
2. Extract to `combined_images/` directory
3. Run the data preprocessing cells in the notebook

### Training

1. Open `DL_project_Resnet.ipynb`
2. Execute cells in order:
   - GPU setup and verification
   - Data loading and preprocessing
   - Model architecture definition
   - Two-phase training execution

### Evaluation

Run the evaluation cells to generate:
- Classification report
- Confusion matrix
- Performance metrics

### Interpretability

Execute the Grad-CAM section to:
- Generate heatmaps for test images
- Compute stability scores
- Visualize model explanations

## 📈 Training Logs

### Phase 1 (Frozen Backbone)
- **Best Validation Accuracy**: 93.2%
- **Training Duration**: ~10 epochs
- **Loss**: Categorical Crossentropy

### Phase 2 (Fine-tuning)
- **Best Validation Accuracy**: 98.6%
- **Training Duration**: ~19 epochs
- **Loss**: Categorical Crossentropy

## 🛠️ Key Technologies

- **TensorFlow/Keras**: Deep learning framework
- **ResNet50**: Pre-trained CNN architecture
- **Grad-CAM**: Interpretability technique
- **OpenCV**: Image processing
- **Scikit-learn**: Evaluation metrics
- **Matplotlib/Seaborn**: Visualization

## 📚 References

- [Dataset Source](Dataset Description and Source Doc.pdf)
- [IEEE Conference Paper](IEEE Conference Paper.pdf)
- ResNet50: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Grad-CAM: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

## 🙏 Acknowledgments

- Alzheimer's MRI dataset providers
- TensorFlow and Keras communities
- Research contributions in interpretable deep learning
