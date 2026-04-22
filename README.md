# Decision Tree & Random Forest - Heart Disease Prediction

##  Overview

This project implements **Decision Tree** and **Random Forest** algorithms on a heart disease dataset to perform classification and compare their performance.

---

## Objective

* Understand tree-based machine learning models
* Train and visualize a Decision Tree
* Prevent overfitting using max depth
* Train a Random Forest model
* Compare model accuracy
* Analyze feature importance
* Evaluate model using cross-validation

---

## Tools & Technologies

* Python
* Pandas
* Scikit-learn
* Matplotlib

---

## Dataset

* Heart Disease Dataset (`heart.csv`)
* Contains medical attributes such as:

  * Age
  * Cholesterol
  * Blood Pressure
  * Heart Rate
* Target:

  * 0 → No disease
  * 1 → Disease present

---

##  How It Works

###  Decision Tree

* Splits data based on feature conditions
* Controlled using `max_depth` to avoid overfitting
* Visualized using tree diagram

---

###  Random Forest

* Ensemble of multiple decision trees
* Improves accuracy and reduces overfitting

---

###  Feature Importance

* Identifies most important features affecting prediction

---

###  Cross Validation

* Evaluates model performance using multiple splits
* Provides reliable accuracy score

---

##  Output

* Decision Tree Accuracy
* Random Forest Accuracy
* Cross-validation scores
* Feature importance graph
* Decision tree visualization

---

##  How to Run

1. Place `heart.csv` in the same folder
2. Run:

   ```bash
   python task5_tree_forest.py
   ```

---

## Result

Random Forest generally performs better than Decision Tree due to reduced overfitting and better generalization.

---

##  Key Concepts

* Decision Tree
* Random Forest
* Overfitting & Underfitting
* Feature Importance
* Cross Validation

---

## Author

Ridhima Panigrahi
