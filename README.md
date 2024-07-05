# Image Recognition to Identify Species of Flowers

## 1.0 Abstract
This report documents the development of a Convolutional Neural Network (CNN) Model using TensorFlow's "tf_flowers" dataset to classify species of flowers. The dataset consists of 3,670 images across five species: Daisy, Dandelion, Roses, Sunflowers, and Tulips. This study focuses on the CNN's model architecture, hyperparameter tuning, and regularization techniques, achieving a final model accuracy of 83%. The report emphasizes the importance of hyperparameter tuning and suggests future research directions.

## 2.0 Introduction
Image classification has been a significant challenge in artificial intelligence, particularly with large datasets. CNNs address this challenge by automating feature extraction, handling large datasets, and recognizing complex patterns. This report explores the use of CNNs to classify flower species within the "tf_flowers" dataset.

## 3.0 Literature Review
Hiary et al. (2018) effectively used CNNs to differentiate between flower species by recognizing nuances in color, shape, and appearance. Their findings support CNNs' high classification accuracy and superior performance over traditional methods, highlighting CNNs' potential in complex image classification tasks.

## 4.0 Methodology

### 4.1 Data Description
The dataset includes 3,670 images divided into five flower species. The data was split into 70% training, 15% validation, and 15% testing sets.

### 4.2 CNN Model

#### 4.2.1 Architecture
The model was built using TensorFlow's Keras API, structured as follows:
- Sequential layers with initial 32 filters of size 3x3 for feature extraction.
- ReLU activation to prevent vanishing gradients and SoftMax for output probabilities.
- MaxPooling of 2x2 to reduce dimensionality.

#### 4.2.2 Regularization Techniques
Dropout rates varied between 0.5 and 0.2, with optional batch normalization to reduce overfitting and stabilize training.

#### 4.2.3 Training and Evaluation
Training used the Adam optimizer over 20 epochs with a batch size of 32. Metrics such as accuracy and loss for both training and validation phases were monitored.

## 5.0 Results

### 5.1 Baseline Model Performance
The initial model with standard parameters achieved 80% accuracy but displayed signs of overfitting.

### 5.2 Experiments on Hyperparameters

| Experiment No. | Learning Rate | Dropout Rate | Batch Size | Batch Normalization | Pooling Type | Accuracy Score |
|----------------|---------------|--------------|------------|---------------------|--------------|----------------|
| 1              | 0.01          | 0.5          | 32         | No                  | MaxPooling   | 80%            |
| 2              | 0.01          | 0.5          | 128        | No                  | MaxPooling   | 71%            |
| 3              | 0.0001        | 0.2          | 32         | No                  | MaxPooling   | 83%            |
| 4              | 0.001         | 0.2          | 128        | Yes                 | MaxPooling   | 80%            |
| 5              | 0.01          | 0.5          | 32         | Yes                 | AveragePooling | 74%          |

**Note:** Experiment 3 provided the best results with an accuracy of 83%.

### 5.3 Visual Insights

- **Figure 1:** Baseline model performance showing initial overfitting.
- **Figure 2:** Improved model performance with adjusted hyperparameters.

## 6.0 Discussion
The best performing model used a learning rate of 0.0001 with a dropout rate of 0.2 and a batch size of 32, achieving 83% accuracy. This configuration effectively balanced the learning process and regularization, enhancing the model's performance.

## 7.0 Conclusion
The series of experiments with the CNN model demonstrated the effectiveness of fine-tuning hyperparameters to enhance performance, achieving a maximum accuracy of 83%. These insights pave the way for future exploration of more advanced models and applications in image classification.

## 8.0 References
1. Hiary, H., et al. (2018). Flower classification using deep convolutional neural networks. *IET Computer Vision, 12*(6), 855-862.
2. Singh, A., & Singh, P. (2020). Image Classification: A Survey. *Journal of Informatics Electrical and Electronics Engineering, 1*(2), 1-9.
