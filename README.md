# Cat vs Dog Image Classification using Transfer Learning

## Project Overview

This project focuses on binary image classification to distinguish between cats and dogs using **Transfer Learning with MobileNetV2**.

A pretrained MobileNetV2 model, trained on ImageNet, is used as a frozen feature extractor. A custom Dense classification layer is added to classify images into two categories: Cat and Dog.

The project demonstrates how transfer learning can be applied to image classification tasks with relatively limited training data.

## Dataset

- **Dataset:** [Cat vs Dog Dataset](https://www.kaggle.com/datasets/anubhab098/cat-vs-dog-dataset)
- **Total Images:** 25,000
- **Cat Images:** 12,503
- **Dog Images:** 12,497

For this project, a balanced subset of 2,000 images was selected:

- 1,000 Cat images
- 1,000 Dog images

### Dataset Split

The selected dataset was divided using stratified sampling.

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 1,400 | 70% |
| Validation | 300 | 15% |
| Testing | 300 | 15% |
| **Total** | **2,000** | **100%** |

## Technologies and Libraries

- Python
- TensorFlow
- TensorFlow Hub
- tf-keras
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- Scikit-learn
- Google Colab
- Kaggle API

## Methodology

### 1. Data Exploration

- Inspected the dataset structure and class distribution.
- Visualized sample images from both categories.
- Examined image dimensions and formats.

### 2. Image Preprocessing

- Randomly selected 1,000 images from each class.
- Converted images to RGB format.
- Resized all images to 224 × 224 pixels.
- Assigned integer labels:
  - Cat = 0
  - Dog = 1
- Normalized pixel values to the range [0, 1].

### 3. Model Architecture

MobileNetV2 was used as a pretrained feature extractor through TensorFlow Hub.

The pretrained layers were frozen, and a Dense output layer with two units was added for classification.

**Architecture:**

Input Image (224 × 224 × 3) → MobileNetV2 Feature Extractor (Frozen) → Feature Vector (1280) → Dense Layer (2) → Cat / Dog

### 4. Model Compilation

| Configuration | Value |
|---|---|
| Base Model | MobileNetV2 |
| Transfer Learning Strategy | Feature Extraction |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Output Units | 2 |
| Epochs | 5 |
| Batch Size | 32 |

### 5. Model Training

The model was trained using the training dataset while monitoring its performance on the validation dataset.

### 6. Model Evaluation

The final model was evaluated using the independent test dataset.

Evaluation metrics included:

- Accuracy
- Loss
- Precision
- Recall
- F1-score
- Confusion Matrix

## Results

### Model Performance

| Metric | Score |
|---|---:|
| Training Accuracy | 99.50% |
| Validation Accuracy | 98.00% |
| Test Accuracy | **98.67%** |
| Test Loss | 0.0356 |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Cat | 99.32% | 98.00% | 98.66% |
| Dog | 98.03% | 99.33% | 98.68% |
| **Overall Accuracy** | | | **98.67%** |

### Confusion Matrix

| Actual / Predicted | Cat | Dog |
|---|---:|---:|
| Cat | 147 | 3 |
| Dog | 1 | 149 |

The model correctly classified 296 out of 300 test images, with only 4 misclassifications.

## Error Analysis

Misclassified images were visualized and examined to understand possible reasons behind incorrect predictions.

Some images contained unusual poses, backgrounds, lighting conditions, or unclear visual features that may have contributed to classification errors.

## Model Saving and Prediction

The trained model was saved in Keras format:

`cat_dog_mobilenetv2.keras`

The saved model was successfully reloaded and tested with a sample image.

Example prediction:

- Cat: 99.81%
- Dog: 0.19%

## Limitations

- Only 2,000 images were used for training and evaluation, rather than the complete dataset.
- Direct image resizing may distort the original aspect ratio.
- The pretrained MobileNetV2 layers remained frozen during training.
- Evaluation was performed using a single random train-validation-test split.

## Future Improvements

- Increase the training dataset size.
- Apply data augmentation techniques.
- Fine-tune selected MobileNetV2 layers.
- Experiment with other pretrained CNN architectures.
- Evaluate generalization using additional unseen datasets.

## Conclusion

This project demonstrates the effectiveness of Transfer Learning for binary image classification using a pretrained MobileNetV2 model.

Despite using a relatively small subset of the available dataset and keeping the pretrained layers frozen, the model achieved **98.67% test accuracy**.

The project provided practical experience with image preprocessing, CNN-based feature extraction, transfer learning, model evaluation, error analysis, and model saving.

