Student Performance Prediction
This project implements a machine learning pipeline to predict student performance based on the "Student-Mat" dataset, which contains various features related to students' personal, social, and academic attributes. The goal is to classify whether a student's final grade (G3) is above or below a threshold (10) using two models: a Neural Network (NN) and a Support Vector Machine (SVM) with an RBF kernel. The pipeline includes data preprocessing, model training with K-Fold cross-validation, performance evaluation, and visualization of results.
Dataset
The dataset used is student-mat.csv, a semicolon-delimited file containing student data from the UCI Machine Learning Repository. It includes features such as age, absences, parental education, study time, and more, with the target variable being the final grade (G3).
Features
Features are categorized as follows:

Numeric: age, absences (standardized using StandardScaler)
Ordinal: Medu, Fedu, traveltime, studytime, failures, famrel, freetime, goout, Dalc, Walc, health (passed through without transformation)
Nominal: Mjob, Fjob, reason, guardian (one-hot encoded using OneHotEncoder)
Binary: school, sex, address, famsize, Pstatus, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic (encoded using OrdinalEncoder)

The target variable (G3) is binarized: grades >= 10 are labeled as 1 (pass), and grades < 10 are labeled as 0 (fail).
Models
Two models are implemented and compared:

Neural Network (NN):

Architecture: Sequential model with 4 layers (256, 128, 64 neurons with ReLU activation, and 1 output neuron with sigmoid activation).
Regularization: L2 regularization applied to all hidden layers.
Optimizer: Adam.
Loss: Binary cross-entropy.
Callbacks: ReduceLROnPlateau to adjust learning rate dynamically.
Training: 5 epochs with a batch size of 32.


Support Vector Machine (SVM):

Kernel: RBF.
Hyperparameter Tuning: Grid search over C (0.1, 1, 10) and gamma (1, 0.1, 0.01) using 3-fold cross-validation.
Probability estimates enabled for ROC curve computation.



Methodology

Preprocessing: A ColumnTransformer is used to preprocess features based on their type (numeric, ordinal, nominal, binary).
Cross-Validation: 10-fold cross-validation is applied to evaluate model performance robustly.
Evaluation Metrics:
Accuracy, Precision, Recall, F1-Score for each fold.
Aggregated confusion matrices for both models.
ROC curves and AUC scores for each fold.
Test error percentages across folds and epochs (for NN).


Visualizations:
Error vs. Fold plot comparing NN and SVM test errors.
Error vs. Epoch plot for NN training and test errors.
ROC curves for both models across all folds.
Confusion matrices for both models.



Requirements
To run the code, install the required Python packages:
pip install pandas numpy scikit-learn tensorflow matplotlib scipy

Usage

Ensure the student-mat.csv dataset is in the same directory as the script.
Run the Python script to:
Preprocess the data.
Train and evaluate the NN and SVM models using 10-fold cross-validation.
Generate and display plots (Error vs. Fold, Error vs. Epoch, ROC Curves, Confusion Matrices).
Print performance metrics and summary results.



Output
The script produces:

Console output with performance metrics for each fold and model.
Visualizations saved as plots (displayed during execution).
A summary indicating which model (NN or SVM) performs better based on average test error.

Notes

The dataset must be semicolon-delimited (;) as specified in the pd.read_csv call.
The NN uses a fixed number of epochs (5) to balance training time and performance; adjust as needed.
The SVM grid search can be computationally intensive; modify the param_grid for faster execution if necessary.
The remove_duplicates function ensures smooth ROC curves by averaging TPR for duplicate FPR values.
All visualizations use Matplotlib and are displayed during execution; save them manually if needed.

License
This project is licensed under the MIT License.
