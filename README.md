Student Performance Prediction
This project implements a machine learning pipeline to predict student performance using the Student-Mat dataset. The goal is to classify whether a student’s final grade (G3) is above or below 10 using two models:

Neural Network (NN)

Support Vector Machine (SVM) with RBF kernel

It includes preprocessing, training with K-Fold cross-validation, performance evaluation, and visualization.

📂 Dataset
File: student-mat.csv

Source: UCI Machine Learning Repository

Delimiter: Semicolon (;)

Target Variable: G3 (final grade)

Binarized Label:

1 → Pass (G3 ≥ 10)

0 → Fail (G3 < 10)

🔍 Features
Type	Columns
Numeric	age, absences (Standardized using StandardScaler)
Ordinal	Medu, Fedu, traveltime, studytime, failures, famrel, freetime, goout, Dalc, Walc, health (Passed through without transformation)
Nominal	Mjob, Fjob, reason, guardian (One-hot encoded using OneHotEncoder)
Binary	school, sex, address, famsize, Pstatus, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic (Encoded using OrdinalEncoder)

🤖 Models
🔹 Neural Network (NN)
Architecture: 4 layers

Hidden Layers: [256, 128, 64 neurons] with ReLU

Output Layer: 1 neuron with Sigmoid

Regularization: L2 on hidden layers

Optimizer: Adam

Loss: Binary Cross-Entropy

Callbacks: ReduceLROnPlateau

Training: 5 epochs, batch size = 32

🔹 Support Vector Machine (SVM)
Kernel: RBF

Tuning: Grid Search over:

C: [0.1, 1, 10]

gamma: [1, 0.1, 0.01]

Cross-validation: 3-fold during tuning

Probability Estimates: Enabled for ROC curves

🔧 Methodology
⚙️ Preprocessing
ColumnTransformer handles different feature types using the encoders/scalers above.

🔁 Cross-Validation
10-fold cross-validation to evaluate both models

📊 Evaluation Metrics
Per fold:

Accuracy

Precision

Recall

F1-score

Visualizations:

ROC Curves (with AUC)

Confusion Matrices

Error vs. Fold (NN vs. SVM)

Error vs. Epoch (NN only)

📈 Visualizations
Error vs. Fold: NN and SVM comparison

Error vs. Epoch: NN training and test errors

ROC Curves: For all folds

Confusion Matrices: Aggregated for both models

✅ Results
Metrics printed per fold for both models

Average test errors computed

Best model highlighted based on lowest average test error

Visuals displayed during execution (can be saved manually)

📦 Requirements
bash
Copy
Edit
pip install pandas numpy scikit-learn tensorflow matplotlib scipy
▶️ Usage
Ensure student-mat.csv is in the same directory as the script.

Run the Python script to:

Preprocess data

Train & evaluate models using 10-fold CV

Generate and show all plots

Print performance metrics

⚠️ Notes
Dataset must be semicolon-delimited (;)

NN uses 5 fixed epochs – adjust as needed

SVM grid search may be computationally expensive

remove_duplicates() smooths ROC curves by averaging TPR for same FPR

Visualizations are shown via Matplotlib

📄 License
This project is licensed under the MIT License.

Let me know if you'd like this exported as a Markdown file or customized for GitHub!
