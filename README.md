# MNIST Handwritten Digit Recognition with PCA

---

## Overview

This project demonstrates handwritten digit recognition on the **MNIST dataset** using **Principal Component Analysis (PCA)** for dimensionality reduction combined with **Logistic Regression**. The focus is on reducing the high-dimensional input space (784 features per image) while maintaining high classification performance.

The workflow includes:  
- Loading and preprocessing the MNIST dataset  
- Standardizing pixel values  
- Dimensionality reduction using PCA  
- Training a Logistic Regression classifier  
- Evaluating and visualizing model performance  
- Analyzing the effect of different numbers of PCA components on accuracy  

---

## Features

- **Dimensionality Reduction:** Reduce 784-dimensional images to a smaller number of components while preserving variance  
- **Classifier Training:** Train logistic regression on PCA-transformed data  
- **Visualization:**  
  - Sample digits  
  - PCA components as images  
  - Cumulative variance explained  
  - 2D PCA projections  
  - Confusion matrix and misclassified examples  
- **Performance Analysis:** Compare training and testing accuracy across different numbers of PCA components  

---

## Requirements

- Python 3.8+  
- Packages:
  ```text
  numpy
  matplotlib
  seaborn
  scikit-learn

  <img width="1444" height="1497" alt="image" src="https://github.com/user-attachments/assets/c76bebda-2361-484c-94ee-e385033f6c03" />

