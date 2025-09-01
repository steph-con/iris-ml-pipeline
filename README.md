# 🌸 End-to-End Machine Learning with the Iris Dataset

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/python-3.13-blue.svg) ![Conda](https://img.shields.io/badge/conda-ready-brightgreen.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ml-blue.svg)

## 📑 Table of Contents

- [Overview](#-project-overview)
- [Results](#-results)
- [Technologies](#️-technologies-used)
- [Installation](#️-installation)
- [Acknowledgements](#-acknowledgements)

This project is an end-to-end implementation of a **machine learning pipeline** using the classic **Iris dataset**.
It walks through **data exploration, visualization, model training, evaluation, and comparison**, and is part of my **data science portfolio**.

---

## 📌 Project Overview

The goal of this project is to demonstrate the **core steps in a machine learning workflow**, applied to the Iris dataset.
This includes:

- Data exploration & visualization
- Preparing data for modeling
- Training and evaluating **six different ML algorithms**
- Comparing performance
- Drawing conclusions and discussing next steps

👉 The main analysis and results can be found in the Jupyter Notebook: **iris_analysis.ipynb**

---

## 📊 Results

- All models performed well, reflecting the clear separability of classes in the Iris dataset.
- **Support Vector Machines (SVM)** achieved the highest accuracy on validation data.
- Linear Discriminant Analysis (LDA) and Logistic Regression also performed strongly.

🔮 **Next Steps**:
Future improvements could include hyperparameter tuning, ensemble methods (Random Forests, Gradient Boosting), and testing on other datasets.

---

## ⚙️ Technologies Used

- Python 3.13
- NumPy
- Pandas
- Matplotlib & Seaborn
- Scikit-learn

---

## ⚙️ Installation

You can run this project using either **Conda** (recommended) or **pip**.

### Option 1: Using Conda (recommended)

```bash
conda env create -f environment.yml
conda activate iris-ml
```

### Option 2: Using pip

```bash
pip install -r requirements.txt
```

---

## 📖 Acknowledgements

This project was inspired by Jason Brownlee’s tutorial: [Machine Learning in Python Step by Step](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/).

While the workflow and structure of this project follow that guide, I wrote the majority of the code independently, with modifications and additions to adapt it into my own style and as a portfolio project.
