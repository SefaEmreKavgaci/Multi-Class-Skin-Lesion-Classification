# Multi-Class Skin Lesion Classification (ISIC) 🩺

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the source code and technical report for my final project in **EEE 443/543 - Neural Networks** at Bilkent University. The goal is to automate the diagnosis of skin cancer by classifying dermoscopic images into **9 distinct categories**, covering both malignant (e.g., Melanoma) and benign (e.g., Nevus) lesions.

## 🧠 Model Architectures

I implemented and compared three state-of-the-art deep learning architectures to evaluate their performance on limited medical data:

1. **EfficientNet-B0** 🏆: The best performer, achieving **70.97% test accuracy** and **81.28% sensitivity** for malignant cases.
2. **ResNet18**: A solid baseline that showed stable generalization but lower peak accuracy (67.16%).
3. **Vision Transformer (ViT-B/16)**: While powerful, it struggled with overfitting due to the small dataset size (approx. 2,357 images), resulting in a large gap between validation and test scores.

## 📂 Project Structure
```
├── data/                   # Dataset folder (Place ISIC images here)
│   ├── actinic keratosis/
│   ├── melanoma/
│   └── ...                 # (other classes)
├── results/                # Outputs (logs, plots, checkpoints)
├── model.py                # Main script (Training & Evaluation pipeline)
├── requirements.txt        # Dependencies
└── Report.pdf              # Detailed technical report
```

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
   git clone https://github.com/sefaemrekavgaci/skin-lesion-classification.git
   cd skin-lesion-classification
```

2. **Install dependencies:**
```bash
   pip install -r requirements.txt
```
   *Note: Create a `requirements.txt` with: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `opencv-python`, `pillow`*

3. **Prepare the data:**
   - Download the **ISIC skin cancer dataset**.
   - Create a folder named `data` in the root directory.
   - Inside `data`, create subfolders for each class name (e.g., `data/melanoma/`, `data/nevus/`) and place the corresponding images there.
   - The script automatically scrapes these folders to build the dataset.

## 💻 Usage

The `model.py` script handles everything using command-line arguments.

### 1. Run the Full Pipeline (Recommended) 🔄
Train all models from scratch, evaluate them, and generate comparison charts:
```bash
python model.py --mode full
```

### 2. Train Only 🏋️‍♂️
Train the models using 5-fold Cross-Validation without running final tests:
```bash
python model.py --mode train
```
This will create a `results/` folder and save the best checkpoints for each fold.

### 3. Evaluate Only 📊
Generate confusion matrices, Grad-CAM visualizations, and classification reports using pre-trained models:
```bash
python model.py --mode evaluate
```

## 📊 Key Results

Detailed analysis is available in `Report.pdf`. Here's a summary of the test set performance:

| Model           | Test Accuracy | Malignant Sensitivity | Param Count |
|-----------------|---------------|----------------------|-------------|
| EfficientNet-B0 | 70.97%        | 81.28%               | 4.0M        |
| ResNet18        | 67.16%        | 76.17%               | 11.2M       |
| ViT-B/16        | 68.22%        | 72.34%               | 85.8M       |

**Insight:** Bigger isn't always better. EfficientNet (4M params) outperformed the massive Vision Transformer (85M params) because it generalizes better on smaller datasets. ViT suffered from a 4.99% generalization gap due to overfitting.

## 🔍 Interpretability (Grad-CAM)

Trust is key in medical AI. I implemented Grad-CAM to visualize where the CNN models are looking:

- Green/Yellow regions indicate areas the model used to make its decision.
- EfficientNet-B0 correctly focused on lesion borders and pigmentation patterns, aligning with the clinical ABCDE rule (Asymmetry, Border, Color, Diameter, Evolving).

## 👤 Author

**Sefa Emre Kavgacı**  
Department of Computer Engineering, Bilkent University  
Student ID: 22103179

---

If you find this project useful, please give it a ⭐!
