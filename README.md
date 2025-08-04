# Comparable Training: CNN Architecture Analysis

A comprehensive comparison of classic CNN architectures (VGG16, AlexNet-inspired, and LeNet-inspired models) on both binary and multi-class image classification tasks.

## 🎯 Project Overview

This project implements and compares three different CNN architectures to analyze their performance across different classification scenarios:

1. **Binary Classification**: Cats vs Dogs (2 classes)
2. **Multi-class Classification**: Sports Recognition (100+ classes)

## 🏗️ Architecture Comparison

### 1. VGG16 (Fine-tuned)
- **Base**: Pre-trained VGG16 from ImageNet
- **Approach**: Transfer learning with frozen base layers
- **Custom Layers**: 
  - Flatten → Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.5) → Output
- **Best For**: Leveraging deep feature representations

### 2. AlexNet-inspired (ResNet50 Base)
- **Base**: Pre-trained ResNet50 (as AlexNet substitute)
- **Approach**: Transfer learning with AlexNet-style dense layers
- **Custom Layers**: 
  - Flatten → Dense(4096) → Dropout(0.5) → Dense(4096) → Dropout(0.5) → Output
- **Best For**: Large-scale feature learning with substantial dense layers

### 3. LeNet-inspired (From Scratch)
- **Base**: Built from scratch
- **Architecture**: 
  - Conv2D(32,5×5) → MaxPool → Conv2D(64,5×5) → MaxPool → Conv2D(128,3×3) → MaxPool
  - Flatten → Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.5) → Output
- **Best For**: Learning task-specific features without pre-training bias

## 📊 Datasets

### Binary Classification
- **Dataset**: Cats vs Dogs
- **Classes**: 2 (cats, dogs)
- **Image Size**: 224×224 pixels
- **Loss Function**: Binary crossentropy
- **Activation**: Sigmoid

### Multi-class Classification
- **Dataset**: Sports Labelling
- **Classes**: 100+ sports categories
- **Image Size**: 224×224 pixels
- **Loss Function**: Categorical crossentropy
- **Activation**: Softmax

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow matplotlib numpy
```

### Running the Code
```python
# Clone and run in Google Colab or local environment
python comparable_training.py
```

### Data Setup
The script expects:
- `cats_and_dogs_filtered.zip` for binary classification
- `Sports_Labelling.zip` for multi-class classification

## 📈 Key Features

- **Reproducible Results**: Fixed random seeds (42) for consistent outcomes
- **Comprehensive Evaluation**: Training/validation accuracy and loss tracking
- **Visual Analysis**: 
  - Training history plots
  - Sample prediction visualizations
  - Model architecture summaries
- **Performance Comparison**: Side-by-side model evaluation
- **Transfer Learning**: Efficient use of pre-trained weights

## 🔧 Technical Details

### Training Configuration
- **Optimizer**: RMSprop (lr=1e-5 for binary, 1e-4 for multi-class)
- **Batch Size**: 32
- **Image Preprocessing**: Normalization (pixel values / 255.0)
- **Epochs**: 10 for binary, 50 for multi-class
- **Validation Split**: 20%

### Model Architecture Insights
```
VGG16:        ~15M parameters (base frozen)
AlexNet-inspired: ~25M parameters (base frozen)  
LeNet-inspired:   ~2M parameters (trainable)
```

## 📊 Results Analysis

The project provides:
- **Accuracy Comparison**: Validation accuracy across all models
- **Loss Tracking**: Training and validation loss curves
- **Prediction Examples**: Visual predictions with confidence scores
- **Performance Metrics**: Final model comparison table

## 🎯 Use Cases

- **Research**: Architecture comparison studies
- **Education**: Understanding CNN design patterns
- **Benchmarking**: Baseline performance establishment
- **Transfer Learning**: Pre-trained vs scratch training analysis

## 🔍 Key Insights

1. **Transfer Learning Advantage**: Pre-trained models (VGG16, ResNet50-based) typically outperform scratch-built models
2. **Task Complexity**: Multi-class classification requires more training epochs and careful tuning
3. **Architecture Trade-offs**: Deeper networks vs computational efficiency
4. **Feature Learning**: Comparison of learned vs hand-crafted features

## 📁 Project Structure

```
comparable_training.py
├── Data Loading & Preprocessing
├── Model Definitions
│   ├── VGG16 Fine-tuned
│   ├── AlexNet-inspired (ResNet50)
│   └── LeNet-inspired (Scratch)
├── Training & Evaluation Pipeline
├── Visualization Tools
└── Prediction Examples
```

