# Neural Network Testing - ARM Dataset Analysis

## Overview

This repository contains a comprehensive comparative analysis of different neural network architectures for time series data classification using the ARM (Activity Recognition from Motion) dataset. The project implements and compares various deep learning approaches including Multi-Layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and ensemble methods.

## Dataset

The project uses the **ARM (Activity Recognition from Motion) dataset**, which contains time series data for human activity recognition. The dataset includes:

- **Training data**: ARM-Metric-train-TS.csv
- **Testing data**: ARM-Metric-test-TS.csv
- **Target variable**: ATYPE (Activity Type) - categorical classification task
- **Features**: Time series metrics from motion sensors

## Project Structure

```
Neural-Nets-practical/
├── README.md                           # This documentation file
├── MLP - ARM.ipynb                    # Multi-Layer Perceptron implementation
├── CNN - ARM.ipynb                    # Convolutional Neural Network implementation
├── RNN - ARM.ipynb                    # Recurrent Neural Network implementation
├── Ensemble MLP - ARM.ipynb           # Ensemble of Multiple MLPs
├── Ensemble MLP-CNN - ARM.ipynb       # Hybrid Ensemble (MLP + CNN)
├── img/                               # Results and visualizations
│   ├── mlp-best-model.png            # MLP model architecture
│   ├── cnn-mnist-best-model.png      # CNN model architecture
│   ├── lstm-best-model.png           # RNN/LSTM model architecture
│   ├── ensemble-mlp-best-model-fAPI.png      # Ensemble MLP results
│   ├── ensemble-mlp-cnn-best-model-fAPI.png  # Hybrid ensemble results
│   └── ...                           # Additional result images
└── old versions/                      # Previous notebook versions
```

## Neural Network Implementations

### 1. Multi-Layer Perceptron (MLP) - ARM.ipynb
- **Architecture**: Feedforward neural network with dense layers
- **Features**: 
  - Data preprocessing and normalization
  - Hyperparameter tuning
  - Dropout for regularization
  - Categorical classification
- **Use case**: Baseline comparison for time series classification

### 2. Convolutional Neural Network (CNN) - ARM.ipynb
- **Architecture**: 2D convolutional layers with pooling
- **Features**:
  - Conv2D layers for feature extraction
  - MaxPooling2D for dimensionality reduction
  - Flatten layer for classification
  - Optimized for time series data structure
- **Use case**: Capturing local temporal patterns in motion data

### 3. Recurrent Neural Network (RNN) - ARM.ipynb
- **Architecture**: Sequential processing with memory
- **Features**:
  - Sequential data processing
  - Temporal dependency modeling
  - LSTM/GRU variants for long-term memory
- **Use case**: Modeling temporal dependencies in activity sequences

### 4. Ensemble MLP - ARM.ipynb
- **Architecture**: Multiple MLP models combined
- **Features**:
  - Model diversity through different architectures
  - Voting or averaging mechanisms
  - Improved generalization and robustness
- **Use case**: Reducing variance and improving prediction stability

### 5. Ensemble MLP-CNN - ARM.ipynb
- **Architecture**: Hybrid ensemble combining MLP and CNN
- **Features**:
  - Complementary model strengths
  - Feature fusion strategies
  - Enhanced representation learning
- **Use case**: Leveraging both local and global patterns

## Technical Requirements

### Dependencies
- **Python**: 3.x
- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Plotly

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Neural-Nets-practical

# Install required packages
pip install tensorflow keras pandas numpy scikit-learn matplotlib
```

## Usage

### Running the Notebooks
1. **Start Jupyter**: `jupyter notebook` or `jupyter lab`
2. **Open desired notebook**: Choose the neural network architecture you want to explore
3. **Execute cells**: Run the cells sequentially to:
   - Load and preprocess the ARM dataset
   - Build and train the neural network
   - Evaluate performance metrics
   - Visualize results

### Data Preprocessing
The notebooks include comprehensive data preprocessing:
- **Loading**: CSV data from GitHub repositories
- **Encoding**: Categorical target variable encoding
- **Normalization**: StandardScaler for feature standardization
- **Validation**: Train/test split and data integrity checks

### Model Training
Each notebook implements:
- **Architecture definition**: Layer-by-layer model construction
- **Compilation**: Loss function, optimizer, and metrics selection
- **Training**: Batch processing with validation
- **Evaluation**: Performance metrics and confusion matrices

## Results and Visualizations

The `img/` directory contains:
- **Model architectures**: Visual representations of network structures
- **Training curves**: Loss and accuracy plots
- **Performance metrics**: Confusion matrices and classification reports
- **Comparison charts**: Cross-model performance analysis

## Key Features

### Comparative Analysis
- **Performance metrics**: Accuracy, precision, recall, F1-score
- **Architecture comparison**: MLP vs CNN vs RNN vs Ensemble
- **Hyperparameter optimization**: Learning rates, layer sizes, dropout rates
- **Computational efficiency**: Training time and resource usage

### Best Practices
- **Regularization**: Dropout layers to prevent overfitting
- **Data augmentation**: Techniques for improving generalization
- **Cross-validation**: Robust model evaluation
- **Hyperparameter tuning**: Systematic parameter optimization

## Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your improvements
4. **Test** thoroughly with the ARM dataset
5. **Submit** a pull request


## License

This project is open source and available under the [MIT License](LICENSE).

## Citation

If you use this code in your research, please cite:
```
@misc{neural-nets-practical,
  title={Neural Network Testing - ARM Dataset Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Neural-Nets-practical}
}
```

---

**Note**: This project is designed for educational and research purposes. The ARM dataset is used to demonstrate various neural network architectures and their applications in time series classification tasks.
