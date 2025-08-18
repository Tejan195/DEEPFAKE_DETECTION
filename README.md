# Analysis and Development of Deepfake Detection

[
[![TensorFlow](https://img.shields.io/badge/

The **"Analysis and Development of Deepfake Detection"** project addresses the escalating threat of AI-generated synthetic media through a **novel hybrid LSTM-GNN-CNN architecture**. As deepfakes proliferate exponentially—from 7,813 instances in 2020 to a projected 8 million by 2025—this research delivers a robust, scalable solution for detecting manipulated media with unprecedented accuracy.

### 🎯 Key Achievements
- **Test Accuracy:** 91.27%
- **AUC Score:** 0.9626
- **Inference Speed:** 0.1s per image (GPU)
- **Model Efficiency:** Optimized for 4-8GB RAM devices
- **Multi-Dataset Validation:** Tested across 22,000+ samples

***

## 🚀 Core Innovations

### **Hybrid Architecture Integration**
- **XceptionNet (CNN):** Spatial feature extraction with 2048-dimensional vectors
- **Graph Neural Networks:** Structural relationship analysis among facial features
- **LSTM Networks:** Temporal consistency evaluation across 10-frame sequences

### **Performance Superiority**
```
Training Progress: 58.8% → 97.96% (40 epochs)
Validation Peak: 89.92% (Epoch 35)
Test Performance: 91.27% accuracy, 0.9626 AUC
Classification: 0.94 precision, 0.91 recall (fake detection)
```

***

## 🎯 Research Objectives

- [x] **Hybrid Model Design:** Integrate spatial, structural, and temporal analysis
- [x] **Enhanced Generalization:** Robust performance across manipulation techniques
- [x] **Resource Optimization:** Efficient pipeline for low-RAM deployment
- [x] **Comprehensive Evaluation:** Multi-metric performance assessment
- [x] **Scalable Platform:** Foundation for real-time and multimodal detection

***

## 🏗️ System Architecture

```mermaid
graph LR
    A[Input Video/Image] --> B[Preprocessing]
    B --> C[MTCNN Face Detection]
    C --> D[XceptionNet CNN]
    D --> E[Feature Extraction 2048-d]
    E --> F[Graph Neural Network]
    F --> G[LSTM Temporal Analysis]
    G --> H[Classification Layer]
    H --> I[Real/Fake Prediction]
```

### **Component Details**

| Component | Function | Specifications |
|-----------|----------|----------------|
| **Preprocessing** | Face alignment, normalization | MTCNN, 299×299 resize, [-1,1] range |
| **XceptionNet** | Spatial feature extraction | 2048-d vectors, ImageNet pretrained |
| **GNN Module** | Structural analysis | 2-layer GCN, ReLU activation, 0.4 dropout |
| **LSTM Module** | Temporal modeling | 192 hidden units, 10-frame sequences |
| **Classification** | Final prediction | SoftMax layer, binary output |

***

## 📊 Dataset Specifications

| Dataset | Samples | Manipulation Types | Usage |
|---------|---------|-------------------|-------|
| **FaceForensics++** | 1,000+ videos | DeepFakes, Face2Face, FaceSwap, NeuralTextures | Training, Robustness Testing |
| **DFDC Preview** | ~5,000 videos | Various synthesis pipelines | Generalization Validation |
| **DFFD** | 15,000+ images | FaceSwap, Face2Face, GAN-based | Fine-tuning, Sensitivity Analysis |
| **DF40** | Mixed samples | Danet, Facedancer, E4S | Next-generation Challenge Testing |

***

## 📈 Performance Metrics

### **Confusion Matrix Results**
```
                 Predicted
Actual    Real    Fake
Real      433     66
Fake      89      914
```

### **Classification Report**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Real** | 0.83 | 0.88 | 0.86 | 499 |
| **Fake** | 0.94 | 0.91 | 0.93 | 1,003 |
| **Weighted Avg** | 0.90 | 0.90 | 0.90 | 1,502 |

### **Training Evolution**
| Epoch | Training Acc | Validation Acc |
|-------|-------------|----------------|
| 1 | 58.8% | 61.74% |
| 10 | 80.59% | 80.71% |
| 20 | 89.34% | 86.29% |
| 35 | 94.46% | **89.92%** |
| 40 | 97.96% | 87.29% |

***

## 🛠️ Installation & Setup

### **Prerequisites**
```bash
# System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 4-8GB RAM (inference) / 16GB+ (training)
- 50GB+ storage for datasets
```

### **Installation Steps**
```bash
# Clone repository
git clone https://github.com/your-repo/deepfake-detection.git
cd deepfake-detection

# Create virtual environment
python -m venv deepfake_env
source deepfake_env/bin/activate  # Linux/Mac
# deepfake_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download datasets
mkdir data
# Place FF++, DFDC, DFFD, DF40 datasets in data/ directory
```

### **Dependencies**
```txt
torch>=1.9.0
torchvision>=0.10.0
tensorflow>=2.6.0
timm>=0.4.12
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
tqdm>=4.62.0
```

***

## 🚀 Usage Guide

### **1. Data Preprocessing**
```bash
python preprocess.py \
    --dataset_path data/ \
    --output_path processed_data/ \
    --image_size 299 \
    --batch_size 32
```

### **2. Model Training**
```bash
python train.py \
    --data_dir processed_data/ \
    --epochs 40 \
    --batch_size 16 \
    --learning_rate 0.00003 \
    --device cuda
```

### **3. Model Evaluation**
```bash
python evaluate.py \
    --model_path models/best_model.pth \
    --test_data processed_data/test/ \
    --output_dir results/
```

### **4. Single Inference**
```bash
python infer.py \
    --input_path path/to/video.mp4 \
    --model_path models/best_model.pth \
    --output_path results/prediction.json
```

### **5. Batch Processing**
```bash
python batch_infer.py \
    --input_dir test_videos/ \
    --model_path models/best_model.pth \
    --output_dir batch_results/
```

***

## 🧪 Experimental Results

### **Comparative Analysis**
| Model Architecture | Accuracy | AUC | Inference Time |
|-------------------|----------|-----|----------------|
| **Our Hybrid Model** | **91.27%** | **0.9626** | **0.1s** |
| CNN-only (XceptionNet) | 87.3% | 0.923 | 0.08s |
| LSTM-only | 82.1% | 0.876 | 0.12s |
| Traditional CNN-LSTM | 88.9% | 0.941 | 0.15s |

### **Ablation Study**
| Component Removal | Accuracy Drop | AUC Drop |
|------------------|---------------|----------|
| Without GNN | -3.2% | -0.031 |
| Without LSTM | -4.8% | -0.047 |
| Without CNN | -12.1% | -0.089 |

***

## 🔧 Configuration

### **Training Configuration**
```python
TRAINING_CONFIG = {
    'epochs': 40,
    'batch_size': 16,
    'learning_rate': 0.00003,
    'weight_decay': 0.0005,
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'early_stopping_patience': 12,
    'gradient_clipping': 1.0
}
```

### **Model Architecture**
```python
MODEL_CONFIG = {
    'cnn_backbone': 'xception',
    'cnn_features': 2048,
    'gnn_layers': 2,
    'gnn_dropout': 0.4,
    'lstm_hidden': 192,
    'lstm_layers': 2,
    'sequence_length': 10,
    'num_classes': 2
}
```

***

## 🚧 Challenges & Solutions

### **Technical Challenges**
| Challenge | Impact | Solution Implemented |
|-----------|--------|---------------------|
| **Class Imbalance** | 2:1 fake:real ratio | Weighted loss, balanced sampling |
| **Dataset Variability** | Quality inconsistencies | Multi-dataset training, augmentation |
| **Memory Constraints** | Limited RAM deployment | Batch processing, model optimization |
| **Temporal Dependencies** | Video sequence modeling | LSTM with gradient clipping |

### **Optimization Strategies**
- **Memory Management:** Batch processing with dynamic sizing
- **Computational Efficiency:** Depthwise separable convolutions
- **Generalization:** Cross-dataset validation protocol
- **Robustness:** Adversarial training techniques

***

## 🔮 Future Roadmap

### **Immediate Enhancements**
- [ ] **Real-time Detection:** Edge device optimization
- [ ] **Multimodal Analysis:** Audio-visual synchronization
- [ ] **Explainable AI:** Attention visualization
- [ ] **Mobile Deployment:** TensorFlow Lite conversion

### **Long-term Vision**
- [ ] **Blockchain Integration:** Media provenance verification
- [ ] **Federated Learning:** Privacy-preserving training
- [ ] **Adversarial Robustness:** Advanced defense mechanisms
- [ ] **Global Standardization:** Unified evaluation protocols

***

## 🏆 Impact & Applications

### **Research Contributions**
- **Novel Architecture:** First LSTM-GNN-CNN hybrid for deepfake detection
- **Performance Benchmark:** State-of-the-art results on multiple datasets
- **Resource Efficiency:** Democratized access through low-RAM optimization
- **Ethical Framework:** Responsible AI principles integration

### **Real-world Applications**
- **Social Media Platforms:** Content verification systems
- **News Organizations:** Journalistic integrity tools
- **Legal Systems:** Evidence authentication
- **Cybersecurity:** Threat detection and prevention

***

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{thapa2025deepfake,
  title={Analysis and Development of Deepfake Detection Using Hybrid LSTM-GNN-CNN Model},
  author={Thapa, Tej Bahadur and Jaypuria, Akash and Mohanty, Amisha and Mohanty, Dibyajyoti and Das, Nilima Rani},
  journal={Advanced AI Research},
  year={2025},
  institution={Siksha 'O' Anusandhan University},
  doi={10.xxxx/xxxx.xxxx}
}
```

***

## 👥 Research Team

| Role | Name | Email | Contribution |
|------|------|-------|-------------|
| **Lead Researcher** | Tej Bahadur Thapa | tejan.thapa.555@gmail.com | Architecture Design, Implementation |
| **ML Engineer** | Akash Jaypuria | akashjaypuria385@gmail.com | Model Training, Optimization |
| **Data Scientist** | Amisha Mohanty | amishamohanty321@gmail.com | Dataset Curation, Analysis |
| **Research Associate** | Dibyajyoti Mohanty | dibyajyotimohanty8812@gmail.com | Evaluation, Validation |
| **Faculty Supervisor** | Asst. Prof. Nilima Rani Das | - | Research Guidance, Review |

**Institution:** Department of Computer Application  
**University:** Siksha 'O' Anusandhan (Deemed to be) University  
**Location:** Bhubaneswar, Odisha, India  
**Project Date:** May 2025

***

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📞 Support

For questions, issues, or collaborations:
- **Email:** tejan.thapa.555@gmail.com
- **Issues:** [GitHub Issues](https://github.com/your-repo/deepfake-detection/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/deepfake-detection/discussions)

***

## 🌟 Acknowledgments

This research builds upon foundational work by:
- Nguyen et al. (2021) - Spatio-temporal feature learning
- Yang et al. (2021) - Texture difference detection
- Jung et al. (2020) - Eye blinking pattern analysis
- Yan et al. (2024) - DF40 dataset contribution

**Special thanks** to the open-source community and dataset contributors who made this research possible.

***

*"Protecting digital authenticity through innovative AI research"*
