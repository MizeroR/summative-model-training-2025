# RealWaste Classification: Traditional ML vs Deep Learning

A comprehensive machine learning pipeline comparing traditional ML and deep learning approaches for waste classification using the RealWaste dataset.

## 📊 Dataset

This project uses the **RealWaste** dataset, an image dataset assembled from waste material received at the Whyte's Gully Waste and Resource Recovery facility in Wollongong, NSW, Australia.

### Dataset Statistics
- **Total Images:** 4,752
- **Number of Classes:** 9
- **Image Source:** Authentic landfill environment
- **Collection Site:** Whyte's Gully Waste and Resource Recovery facility, Wollongong, NSW, Australia

### Class Distribution
| Class | Image Count | Percentage |
|-------|-------------|------------|
| Plastic | 921 | 19.38% |
| Metal | 790 | 16.62% |
| Paper | 500 | 10.52% |
| Miscellaneous Trash | 495 | 10.42% |
| Cardboard | 461 | 9.70% |
| Vegetation | 436 | 9.18% |
| Glass | 420 | 8.84% |
| Food Organics | 411 | 8.65% |
| Textile Trash | 318 | 6.69% |

**Note:** The above labeling may be further subdivided as required (e.g., Transparent Plastic, Opaque Plastic).

### Citation

If you use the RealWaste dataset in your work, please cite the original paper:
```bibtex
@article{majchrowska2023realwaste,
  title={RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning},
  journal={Information},
  volume={14},
  number={12},
  pages={633},
  year={2023},
  publisher={MDPI},
  url={https://www.mdpi.com/2078-2489/14/12/633}
}
```

### License

This dataset is licensed under **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International).

- ✅ You may use this dataset for **research and educational purposes**
- ✅ You must give **appropriate credit** to the original authors
- ✅ You must **share adaptations** under the same license
- ❌ You may **not use** this dataset for **commercial purposes**

For more information, visit: [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 🎯 Project Overview

This notebook demonstrates a complete machine learning pipeline that:

1. **Compares traditional ML and deep learning approaches** for image classification
2. **Implements 7 different models** across both paradigms
3. **Conducts comprehensive experiments** with proper train/validation/test splits
4. **Analyzes results** including per-class performance, confusion patterns, and model confidence
5. **Provides actionable insights** into dataset limitations and recommendations

## 🧪 Models Implemented

### Traditional Machine Learning (Scikit-learn)
Uses pre-trained MobileNetV2 for feature extraction:
1. **Random Forest Classifier** - Ensemble of 200 decision trees
2. **Support Vector Machine (SVM)** - RBF kernel with optimized parameters
3. **Gradient Boosting Classifier** - Sequential ensemble learning

### Deep Learning (TensorFlow)

#### Sequential API
4. **Simple CNN from Scratch** - Custom 3-block convolutional architecture
5. **MobileNetV2 Transfer Learning** - Pre-trained on ImageNet, frozen base

#### Functional API
6. **Multi-Branch CNN** - Parallel 3x3 and 5x5 convolutional paths
7. **EfficientNetB0 with Fine-tuning** - Two-phase training with layer unfreezing

## 🚀 Key Features

- ✅ **Automated dataset downloading** from UCI Machine Learning Repository
- ✅ **Data augmentation** (rotation, shifts, zoom, flips) for improved generalization
- ✅ **Stratified train/validation/test split** (70/15/15) maintaining class distribution
- ✅ **tf.data API** integration via ImageDataGenerator
- ✅ **Callbacks**: Early stopping, learning rate reduction
- ✅ **Comprehensive visualizations**: 15+ plots including confusion matrices, training histories, per-class metrics
- ✅ **Error analysis**: Misclassification patterns and model confidence analysis
- ✅ **Class imbalance handling**: Detection and recommendations
- ✅ **Production-ready outputs**: Saved models and results for deployment

## 📋 Requirements
```python
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
```

## 🏃 Quick Start

### Option 1: Google Colab (Recommended)
1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Enable GPU: `Runtime → Change runtime type → GPU`
3. Run all cells: `Runtime → Run all`

### Option 2: Kaggle
1. Upload notebook to [Kaggle](https://www.kaggle.com/)
2. Enable GPU: `Settings → Accelerator → GPU T4 x2`
3. Add RealWaste dataset or let the notebook download it automatically
4. Run all cells

### Option 3: Local Environment
```bash
# Clone repository
git clone https://github.com/yourusername/realwaste-classification.git
cd realwaste-classification

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook realwaste_classification.ipynb
```

## 📊 Results Summary

| Model Type | Best Model | Test Accuracy |
|------------|-----------|---------------|
| Traditional ML | Random Forest | ~65% |
| Deep Learning | EfficientNetB0 | ~80-90% |

**Key Finding:** Deep learning models significantly outperform traditional ML approaches, demonstrating the importance of end-to-end feature learning for complex visual classification tasks in authentic landfill environments.

## 🔍 Key Insights

1. **Class Imbalance (2.90x ratio)**: The dataset reflects real-world waste distribution with plastic being most common
2. **Transfer Learning Effectiveness**: Pre-trained models improve accuracy by ~15-20% over CNNs from scratch
3. **Challenging Categories**: Textile Trash and Food Organics show lower F1-scores due to limited samples and visual similarity
4. **Landfill Environment Challenges**: Variable lighting, occlusion, and contamination make classification inherently difficult
5. **Model Confidence**: Low-confidence predictions (<0.5) should undergo human verification in production

## 📁 Output Files

The notebook generates the following outputs in `/kaggle/working/` or your local directory:

- `model_comparison_results.csv` - Performance metrics for all 7 models
- `per_class_performance.csv` - Precision, recall, F1-score per waste category
- `confusion_patterns.csv` - Top misclassification patterns
- `training_history.csv` - Epoch-by-epoch training metrics
- `best_realwaste_model.h5` - Trained model weights (best performer)

## 🎓 Educational Value

This project demonstrates:
- ✅ Complete ML pipeline from data loading to deployment
- ✅ Comparison of traditional ML vs deep learning paradigms
- ✅ Proper experimental methodology with train/val/test splits
- ✅ TensorFlow Sequential and Functional APIs
- ✅ Transfer learning and fine-tuning techniques
- ✅ Critical analysis of model performance and dataset limitations
- ✅ Professional documentation and visualization

## ⚠️ Important Notes

### Dataset Usage
- **Academic/Research Use Only** - This dataset is licensed for non-commercial purposes
- **Proper Attribution Required** - Always cite the original RealWaste paper
- **Share-Alike** - Derivative works must use the same CC BY-NC-SA 4.0 license

### Model Limitations
- Trained on Australian landfill waste - may not generalize to other regions
- Class imbalance may cause bias toward common categories (Plastic, Metal)
- Real-world deployment requires confidence thresholding and human verification
- Model size (~100-300MB) may require optimization for edge deployment

## 🔧 Customization

### Adjust Image Resolution
```python
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Reduce for faster training
```

### Modify Training Parameters
```python
BATCH_SIZE = 16  # Adjust based on GPU memory
EPOCHS = 20      # Reduce for faster experimentation
```

### Add New Models
```python
# Example: Add ResNet50
from tensorflow.keras.applications import ResNet50

base = ResNet50(include_top=False, weights='imagenet')
# ... add classification head
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Ensure code follows PEP 8 style guidelines
4. Add documentation for new features
5. Submit a pull request

## 📧 Contact

For questions or collaboration:
- **Author:** [Your Name]
- **GitHub:** [@MizeroR](https://github.com/MizeroR)

## 🙏 Acknowledgments

- **RealWaste Dataset Authors** - For providing this valuable real-world dataset
- **Whyte's Gully Facility** - For enabling data collection in an authentic landfill environment
- **TensorFlow Team** - For excellent deep learning framework
- **Scikit-learn Contributors** - For robust machine learning tools

## 📄 License

- **Code:** MIT License (see `LICENSE` file)
- **Dataset:** CC BY-NC-SA 4.0 (see dataset citation above)

---

**Disclaimer:** This project is for educational and research purposes only. The RealWaste dataset license (CC BY-NC-SA 4.0) prohibits commercial use. Always verify waste classification results before operational deployment in waste management systems.