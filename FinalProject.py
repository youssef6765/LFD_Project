import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.interpolate import interp1d
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def remove_duplicates(x, y):
    """Remove duplicates from x while averaging corresponding y values."""
    unique_x, indices = np.unique(x, return_index=True)
    unique_y = [np.mean(y[np.where(x == val)]) for val in unique_x]
    return unique_x, np.array(unique_y)



# Load dataset
data = pd.read_csv('student-mat.csv', delimiter=';')

'''
Data preprocessing:
1-numeric:defines a list of numeric feature columns in the dataset that will undergo scaling or preprocessing for the model.
2-ordinal:defines a list of ordinal feature columns in the dataset, where the values represent ranked or ordered data that can be used directly or encoded for the model.
3-nominal:defines a list of nominal (categorical) feature columns in the dataset, where the values represent categories without any inherent order  
4-binary: defines a list of binary feature columns in the dataset, where each feature has only two possible values (e.g., yes/no, true/false), often encoded as 0 and 1 for modeling
'''
# Define features and target
numeric = ['age', 'absences']
ordinal = ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']
nominal = ['Mjob', 'Fjob', 'reason', 'guardian']
binary = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

target = 'G3'
features = data.drop(columns=[target])
labels = data[target]

# Binarized the target for classification
labels = (labels >= 10).astype(int)

# Create preprocessing pipeline
full_pipeline = ColumnTransformer([
    ("binary", OrdinalEncoder(), binary),
    ("nominal", OneHotEncoder(), nominal),
    ("num", StandardScaler(), numeric),
    ("ordinal", 'passthrough', ordinal)
])

# Apply pipeline to features
X_processed = full_pipeline.fit_transform(features)
y = labels

# Define K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

def create_neural_model():
    model = Sequential([
        Dense(256, activation='relu', kernel_regularizer='l2', input_shape=(X_processed.shape[1],)),
        Dense(128, activation='relu', kernel_regularizer='l2'),#hidden_layer1
        Dense(64, activation='relu', kernel_regularizer='l2'),#hidden_layer2
        Dense(1, activation='sigmoid')#output
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Metrics storage
nn_fold_test_errors, svm_fold_test_errors = [], []
nn_roc_data, svm_roc_data = [], []
nn_measures, svm_measures = [], []
nn_all_true_labels = []
nn_all_pred_labels = []
svm_all_true_labels = []
svm_all_pred_labels = []
nn_epoch_train_errors = []
nn_epoch_test_errors = []

for fold, (train_index, test_index) in enumerate(kf.split(X_processed), 1):
    X_train, X_test = X_processed[train_index], X_processed[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Neural Network
    nn_model = create_neural_model()
    #early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

    nn_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                 epochs=5, batch_size=32, callbacks=[reduce_lr], verbose=0)

    nn_test_pred = (nn_model.predict(X_test).flatten() >= 0.5).astype(int)
    nn_fold_test_errors.append((1 - accuracy_score(y_test, nn_test_pred)) * 100)
    nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_model.predict(X_test).flatten())
    nn_roc_data.append((nn_fpr, nn_tpr, auc(nn_fpr, nn_tpr)))


    nn_accuracy = accuracy_score(y_test, nn_test_pred)
    nn_precision = precision_score(y_test, nn_test_pred)
    nn_recall = recall_score(y_test, nn_test_pred)
    nn_f1 = f1_score(y_test, nn_test_pred)
    nn_measures.append((nn_accuracy, nn_precision, nn_recall, nn_f1))

    nn_all_true_labels.extend(y_test)
    nn_all_pred_labels.extend(nn_test_pred)


    # SVM with RBF Kernel and Grid Search
    param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}
    svm_model = GridSearchCV(SVC(probability=True), param_grid, cv=3, verbose=0)
    svm_model.fit(X_train, y_train)

    svm_test_pred = svm_model.predict(X_test)
    svm_fold_test_errors.append((1 - accuracy_score(y_test, svm_test_pred)) * 100)
    svm_probs = svm_model.predict_proba(X_test)[:, 1]
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
    svm_roc_data.append((svm_fpr, svm_tpr, auc(svm_fpr, svm_tpr)))


    svm_accuracy = accuracy_score(y_test, svm_test_pred)
    svm_precision = precision_score(y_test, svm_test_pred)
    svm_recall = recall_score(y_test, svm_test_pred)
    svm_f1 = f1_score(y_test, svm_test_pred)
    svm_measures.append((svm_accuracy, svm_precision, svm_recall, svm_f1))


    svm_test_pred = svm_model.predict(X_test)
    svm_all_true_labels.extend(y_test)
    svm_all_pred_labels.extend(svm_test_pred)

nn_epoch_train_errors = []
nn_epoch_test_errors = []

# Loop through 5 epochs
for epoch in range(5):
    # Train for one epoch
    history = nn_model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    
    # Compute training error
    train_pred = (nn_model.predict(X_train).flatten() >= 0.5).astype(int)
    train_error = (1 - accuracy_score(y_train, train_pred)) * 100
    nn_epoch_train_errors.append(train_error)
    
    # Compute test error
    test_pred = (nn_model.predict(X_test).flatten() >= 0.5).astype(int)
    test_error = (1 - accuracy_score(y_test, test_pred)) * 100
    nn_epoch_test_errors.append(test_error)


# Plot Error vs Fold
folds = range(1, 11)
plt.plot(folds, nn_fold_test_errors, label="NN Test Error", marker='o')
plt.plot(folds, svm_fold_test_errors, label="SVM Test Error", marker='x')
plt.xlabel("Fold")
plt.ylabel("Error (%)")
plt.title("Error Across Folds")
plt.legend()
plt.show()



# Plot errors vs epochs
epochs = range(1, 6)  # Epochs are 1 through 5
plt.figure(figsize=(8, 6))
plt.plot(epochs, nn_epoch_train_errors, label="Training Set Error (%)", marker='o')
plt.plot(epochs, nn_epoch_test_errors, label="Test Set Error (%)", marker='x')
plt.xlabel("Epoch (Training set / Test set)")
plt.ylabel("Error (%)")
plt.title("Model Error vs Epoch")
plt.legend()
plt.grid(True)
plt.show()


# Plot NN ROC curves
for fold, (fpr, tpr, roc_auc) in enumerate(nn_roc_data, 1):
    fpr, tpr = remove_duplicates(fpr, tpr)  # Remove duplicates
    fpr_smooth = np.linspace(0, 1, 500)  # Interpolation range
    tpr_interp = interp1d(fpr, tpr, kind='cubic')(fpr_smooth)  # Cubic interpolation
    plt.plot(fpr_smooth, tpr_interp, label=f"NN Fold {fold} (AUC = {roc_auc:.2f})")

# Plot SVM ROC curves
for fold, (fpr, tpr, roc_auc) in enumerate(svm_roc_data, 1):
    fpr, tpr = remove_duplicates(fpr, tpr)  # Remove duplicates
    fpr_smooth = np.linspace(0, 1, 500)  # Interpolation range
    tpr_interp = interp1d(fpr, tpr, kind='cubic')(fpr_smooth)  # Cubic interpolation
    plt.plot(fpr_smooth, tpr_interp, linestyle='--', label=f"SVM Fold {fold} (AUC = {roc_auc:.2f})")


# Plot settings
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()



print("\nPerformance Measures for Neural Network:")
for fold, (accuracy, precision, recall, f1) in enumerate(nn_measures, 1):
    print(f"Fold {fold}: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1-Score={f1:.2f}")

print("\nPerformance Measures for SVM:")
for fold, (accuracy, precision, recall, f1) in enumerate(svm_measures, 1):
    print(f"Fold {fold}: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1-Score={f1:.2f}")

# Print confusion matrix for Neural Network
print("\nConfusion Matrix for Neural Network (Aggregated):")
nn_conf_matrix = confusion_matrix(nn_all_true_labels, nn_all_pred_labels)
print(nn_conf_matrix)

# Visualize confusion matrix for Neural Network
ConfusionMatrixDisplay(confusion_matrix=nn_conf_matrix, display_labels=[0, 1]).plot()
plt.title("Neural Network Confusion Matrix")
plt.show()

# Print confusion matrix for SVM
print("\nConfusion Matrix for SVM (Aggregated):")
svm_conf_matrix = confusion_matrix(svm_all_true_labels, svm_all_pred_labels)
print(svm_conf_matrix)

# Visualize confusion matrix for SVM
ConfusionMatrixDisplay(confusion_matrix=svm_conf_matrix, display_labels=[0, 1]).plot()
plt.title("SVM Confusion Matrix")
plt.show()

# Summary
print("\nNeural Network Average Test Error: {:.2f}%".format(np.mean(nn_fold_test_errors)))
print("SVM Average Test Error: {:.2f}%".format(np.mean(svm_fold_test_errors)))
print("\nBest Model Based on Results:")
if np.mean(nn_fold_test_errors) < np.mean(svm_fold_test_errors):
    print("Neural Network")
else:
    print("SVM")
