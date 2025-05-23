Analysis and Development of Deepfake Detection
Overview
The "Analysis and Development of Deepfake Detection" project focuses on designing a novel system to detect deepfake media using a hybrid LSTM-GNN-CNN model. Deepfakes, powered by advancements in deep learning and generative adversarial networks (GANs), pose significant threats to digital authenticity, including identity fraud, misinformation, and cyberattacks. This project aims to combat these threats by developing a scalable, generalizable solution that enhances digital trust and security.
The system leverages XceptionNet for spatial feature extraction, Graph Neural Networks (GNNs) for structural relationship analysis, and Long Short-Term Memory (LSTM) networks for temporal consistency evaluation. It achieves a test accuracy of 89.68% and an AUC of 0.9566 across diverse datasets, demonstrating its effectiveness in real-world scenarios.
Objectives

Design and implement a hybrid LSTM-GNN-CNN model for accurate deepfake detection.
Enhance generalization across various deepfake generation techniques.
Streamline detection with an efficient preprocessing pipeline for low-RAM devices.
Evaluate model performance using comprehensive metrics (accuracy, AUC, precision, recall, F1-score).
Create a scalable platform for future advancements in deepfake detection.

Methodology
The project employs a hybrid approach integrating three key components:

CNN (XceptionNet): Extracts spatial features from images, identifying anomalies like texture inconsistencies.
GNN: Analyzes structural relationships among facial features, detecting non-local inconsistencies.
LSTM: Captures temporal dynamics in video sequences, identifying unnatural patterns like jittery expressions.

System Workflow

Preprocessing: Aligns faces using MTCNN, resizes images to 299x299, and normalizes them.
Feature Extraction: Uses XceptionNet to extract 2048-dimensional feature vectors.
Structural Analysis: Applies GNN to model facial relationships as graphs.
Temporal Analysis: Uses LSTM to analyze sequences of 10 frames for consistency.
Classification: Outputs real/fake probabilities via a SoftMax layer.

Datasets
The model was trained and evaluated on the following datasets:

FaceForensics++ (FF++): Over 1,000 video sequences with manipulations (DeepFakes, Face2Face, FaceSwap, NeuralTextures).
DeepFake Detection Challenge (DFDC) Preview: ~5,000 videos with diverse actors and lighting conditions.
DFFD: Over 15,000 manipulated images with varying severity levels.
DF40: A custom dataset combining Danet, Facedancer, and E4S data, representing next-generation deepfake challenges [Yan et al., 2024].

Results

Training Accuracy: Improved from 58.80% (Epoch 1) to 95.12% (Epoch 40).
Validation Accuracy: Peaked at 89.09% (Epoch 35).
Test Performance: Achieved 89.68% accuracy and 0.9566 AUC on a test set of 1502 samples (499 real, 1003 fake).
Scalability: Efficiently handles large datasets on low-RAM devices, with an average inference time of 0.1 seconds per image on a GPU.

Requirements
Hardware

Training: Server with GPU support (e.g., NVIDIA GPU with CUDA), 16 GB RAM, 50 GB storage.
Inference: Low-RAM devices (e.g., laptops with 4-8 GB RAM).

Software

OS: Windows, Linux, or macOS.
Frameworks/Libraries: Python 3.8+, TensorFlow, PyTorch, TIMM, NumPy, OpenCV, Matplotlib.
Tools: Jupyter Notebook, VS Code, Git.

Installation

Clone the repository:git clone https://github.com/your-repo/deepfake-detection.git


Install dependencies:pip install -r requirements.txt


Download datasets (FF++, DFDC, DFFD, DF40) and place them in the data/ directory.

Usage

Preprocess the dataset:python preprocess.py --dataset_path data/ --output_path processed_data/


Train the model:python train.py --data_dir processed_data/ --epochs 40 --batch_size 16


Evaluate the model:python evaluate.py --model_path models/best_model.pth --test_data processed_data/test/


Run inference on a new video/image:python infer.py --input_path path/to/your/video.mp4 --output_path results/



Future Work

Develop lightweight models for real-time detection on edge devices.
Incorporate multimodal analysis (e.g., audio, behavioral cues).
Integrate explainable AI (XAI) for interpretable results.
Explore blockchain-based media provenance for authenticity verification.

Acknowledgments
This project builds upon research and datasets from the following works:

Nguyen XH, et al. "Learning spatio-temporal features to detect manipulated facial videos created by the deepfake techniques." Forensic Sci Int: Digital Invest, 2021.
Yang J, et al. "Detecting fake images by identifying potential texture difference." Future Gener Comput Syst, 2021.
Taeb M, Chi H. "Comparison of deepfake detection techniques through deep learning." J Cybersecur Privacy, 2022.
Stroebel L, et al. "A systematic literature review on the effectiveness of deepfake detection techniques." J Cyber Security Technology, 2023.
Lee S, et al. "Detecting handcrafted facial image manipulations and GAN-generated facial images using shallow-FakeFaceNet." Applied Soft Computing, 2021.
Guarnera L, et al. "Fighting deepfake by exposing the convolutional traces on images." IEEE Access, 2020.
Jung T, et al. "DeepVision: Deepfakes detection using human eye blinking pattern." IEEE Access, 2020.
Yan, Zhiyuan, et al. "DF40: Toward Next-Generation Deepfake Detection." arXiv preprint arXiv:2406.13495, 2024.

Authors

Tej Bahadur Thapa (tejan.thapa.555@gmail.com)
Akash Jaypuria (akashjaypuria385@gmail.com)
Amisha Mohanty (amishamohanty321@gmail.com)
Dibyajyoti Mohanty (dibyajyotimohanty8812@gmail.com)
Asst. Prof. Nilima Rani Das

Institution: Department of Computer Application, Siksha 'O' Anusandhan (Deemed to be) University, Bhubaneswar, Odisha, India
Date: May 23, 2025
