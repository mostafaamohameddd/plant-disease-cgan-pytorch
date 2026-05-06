# Synthetic Plant Leaf Generation via DCGAN and CGAN 

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Generative_AI-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

##  Project Overview
[cite_start]Plant disease analysis and classification models often suffer from limited and highly imbalanced training data[cite: 12]. [cite_start]This project implements a complete, end-to-end generative modeling pipeline in **PyTorch** to synthesize realistic plant leaf images using the **Plant Village dataset**[cite: 5]. 

The pipeline develops two distinct generative systems from scratch:
1. [cite_start]**Standard DCGAN:** Learns the underlying image distribution of specific plant families (e.g., Apple or Tomato) to generate realistic leaf textures and structures[cite: 7].
2. [cite_start]**Conditional GAN (CGAN):** An advanced architecture conditioned on leaf health state, enabling the highly controlled synthesis of purely "Healthy" or "Diseased" leaves[cite: 8].

##  Technical Highlights & Engineering Best Practices
* [cite_start]**Custom Architectures:** Implemented DCGAN-style Generators and Discriminators utilizing transposed convolutions, Batch Normalization, and LeakyReLU activations[cite: 74, 76, 77, 79].
* [cite_start]**Controlled Optimization:** Stabilized early-stage training and mitigated adversarial overpowering by enforcing specific learning rate ratios (**Discriminator LR: 0.0004**, **Generator LR: 0.0001**)[cite: 98, 99, 100].
* [cite_start]**Mode Collapse Mitigation:** Engineered a programmatic checkpoint and diversity evaluation at Epoch 10 to monitor training dynamics and proactively detect mode collapse risks[cite: 29, 106, 107].
* [cite_start]**Conditional Label Embeddings:** Extended the standard GAN into a CGAN by learning spatial label embeddings (`1x64x64`) in the Discriminator and utilizing channel-wise concatenation in the Generator for strict class-guided synthesis[cite: 34, 88, 91, 92].
* [cite_start]**Reproducibility:** Maintained strict pipeline reproducibility through environmental seed control, explicit dependencies, and deterministic `64x64` normalized preprocessing[cite: 49, 66, 69].

##  Repository Structure
```text
plant-disease-cgan-pytorch/
├── docs/                               
│   └── Technical_Report.pdf            # Comprehensive breakdown of architecture, training dynamics, and evaluation
├── notebooks/                          
│   └── plant_disease_synthesis.ipynb   # Core PyTorch implementation (Data pipeline, Model definitions, Training loop)
├── .gitignore                          # Excludes large datasets and cache files
├── README.md                           # Project documentation
└── requirements.txt                    # Environment dependencies
```

## Model Inference & Usage

1. Installation
Clone the repository and install the required dependencies:
git clone [https://github.com/yourusername/plant-disease-cgan-pytorch.git](https://github.com/yourusername/plant-disease-cgan-pytorch.git)
cd plant-disease-cgan-pytorch
pip install -r requirements.txt

2. Controllable Generation (Inference)
The project provides a dedicated inference utility allowing users to dynamically request synthetic data generation based on specific clinical categories.
# Initialize inference for the Conditional GAN
# Ask the model to generate 8 synthetic "diseased" leaves
generate_leaf_type(
    cgen=cgan_gen, 
    leaf_type="diseased", 
    z_dim=100, 
    n_samples=8
)



## Training Results
The generative models demonstrated clear visual evolution across 50 epochs. 
The generated outputs successfully transitioned from unstructured, high-variance noise in early epochs to forming coherent leaf-like structures, distinct textures, and accurate color consistencies mimicking real agricultural data.
Furthermore, the Conditional GAN achieved strong condition separability, accurately outputting targeted "Healthy" or "Diseased" leaf samples upon request.
