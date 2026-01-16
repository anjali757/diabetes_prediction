#  Diabetes Prediction Using Machine Learning (SVM)

##  Project Overview

This project predicts whether a person is **diabetic or not** using medical data and **Machine Learning**.
It uses the **Support Vector Machine (SVM)** algorithm to classify patients based on health parameters.
The model is trained on a diabetes dataset and gives accurate predictions for new patient data.

##  Objective

* Analyze diabetes-related health data
* Clean and preprocess missing values
* Standardize features for better model performance
* Train an ML model to predict diabetes
* Build a simple predictive system for real-world input


##  Technologies & Libraries Used

* **Python**
* **NumPy** – numerical operations
* **Pandas** – data analysis
* **Matplotlib & Seaborn** – data visualization
* **Scikit-learn** – ML model, preprocessing & evaluation


## Dataset Features

The dataset contains the following medical attributes:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

**Target Variable:**

* `Outcome`

  * `0` → Non-Diabetic
  * `1` → Diabetic



##  Project Workflow

1. Load and explore the dataset
2. Handle missing values by replacing zeros with:

   * Mean (Glucose, Blood Pressure)
   * Median (BMI, Skin Thickness, Insulin)
3. Perform data visualization:

   * Histograms
   * Pair plots
   * Correlation heatmap
4. Standardize features using **StandardScaler**
5. Split data into training and testing sets
6. Train **Support Vector Machine (Linear Kernel)**
7. Evaluate model using accuracy score
8. Build a predictive system for new inputs


##  Model Performance

* **Algorithm:** Support Vector Machine (SVM – Linear Kernel)
* **Training Accuracy:** High (model fits well)
* **Testing Accuracy:** Reliable and consistent

(Exact accuracy may vary depending on random split)


##  Example Prediction

```python
Input:
(4,110,92,0,37.6,0,191,30)

Output:
Person is Diabetic
```

The model standardizes the input and predicts whether the person has diabetes.


##  How to Run the Project

1. Clone the repository
2. Install required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. Run the Python script or Jupyter Notebook
4. Enter patient details to get diabetes prediction


##  Future Improvements

* Use more advanced models (Random Forest, XGBoost)
* Add cross-validation
* Handle class imbalance
* Deploy as a web app using Flask or Streamlit




