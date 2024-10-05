# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt

# Load the dataset using the file paths
X_train_path = r"C:\Users\tomin\OneDrive\Machine Learning\Final Project\X_train.txt"
y_train_path = r"C:\Users\tomin\OneDrive\Machine Learning\Final Project\y_train.txt"
X_test_path = r"C:\Users\tomin\OneDrive\Machine Learning\Final Project\X_test.txt"
y_test_path = r"C:\Users\tomin\OneDrive\Machine Learning\Final Project\y_test.txt"
activity_labels_path = r"C:\Users\tomin\OneDrive\Machine Learning\Final Project\activity_labels.txt"

# Load the datasets
X_train = pd.read_csv(X_train_path, header=None, delimiter=r'\s+')
y_train = pd.read_csv(y_train_path, header=None, delimiter=r'\s+').values.ravel()
X_test = pd.read_csv(X_test_path, header=None, delimiter=r'\s+')
y_test = pd.read_csv(y_test_path, header=None, delimiter=r'\s+').values.ravel()

# Load activity labels
activity_labels = pd.read_csv(activity_labels_path, header=None, delimiter=' ', index_col=0, names=['Activity'])
activity_names = dict(zip(activity_labels.index, activity_labels['Activity']))

# Map activity labels to the y data
y_train = pd.Series(y_train).map(activity_names)
y_test = pd.Series(y_test).map(activity_names)

# Convert categorical labels to integer indices
activity_name_to_index = {name: index for index, name in enumerate(y_train.unique())}
y_train_int = pd.Series(y_train).map(activity_name_to_index)
y_test_int = pd.Series(y_test).map(activity_name_to_index)

# Preprocessing pipeline for X_train and X_test
numeric_features = X_train.select_dtypes(include=['float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Define preprocessing steps for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing to X_train and X_test
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Ensure alignment between X_train_preprocessed and y_train_int
print(f"Before alignment: X_train_preprocessed shape: {X_train_preprocessed.shape}, y_train_int shape: {y_train_int.shape}")
y_train_int = y_train_int[:X_train_preprocessed.shape[0]]  # Align if necessary
print(f"After alignment: X_train_preprocessed shape: {X_train_preprocessed.shape}, y_train_int shape: {y_train_int.shape}")

# Split dataset into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_preprocessed, y_train_int, test_size=0.2, random_state=42)

# Define a function to create the stacking model
def get_stacking_model():
    # Define base models
    level0 = list()
    level0.append(('perceptron', Perceptron(max_iter=1000, tol=1e-3)))
    level0.append(('random_forest', RandomForestClassifier(n_estimators=50, random_state=42)))  # Reduced number of trees
    level0.append(('svm', SVC(kernel='rbf', C=1.0, random_state=42)))

    # Define meta learner model
    level1 = LogisticRegression(solver='saga', max_iter=500, tol=1e-3, C=0.1)  # Reduced max_iter

    # Define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)  # Reduced number of folds
    return model

# Define the models to evaluate
def get_models():
    models = dict()
    models['perceptron'] = Perceptron(max_iter=1000, tol=1e-3)
    models['random_forest'] = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced number of trees
    models['svm'] = SVC(kernel='rbf', C=1.0, random_state=42)
    models['stacking'] = get_stacking_model()
    return models

# Evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)  # Reduced number of folds and repeats
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# Train and evaluate models
models = get_models()
results, names = list(), list()
for name, model in models.items():
    # Train and evaluate each model
    model.fit(X_train_split, y_train_split)
    y_pred = model.predict(X_val_split)
    acc = accuracy_score(y_val_split, y_pred)
    print(f"{name.capitalize()} Accuracy: {acc}")
    print(f"\nClassification Report for {name.capitalize()}:")
    print(classification_report(y_val_split, y_pred))
    print(f"\nPerforming Cross-Validation for {name.capitalize()}...")
    scores = evaluate_model(model, X_train_split, y_train_split)
    results.append(scores)
    names.append(name)
    print(f"{name.capitalize()} Cross-Validation Accuracies: {scores}")
    print(f"{name.capitalize()} Mean Accuracy: {scores.mean()}")

# Plot model performance for comparison
plt.figure(figsize=(10, 6))
plt.boxplot(results, tick_labels=names, showmeans=True)
plt.title('Model Performance Comparison')
plt.ylabel('Accuracy')
plt.show()

# CNN Model
def create_cnn_model():
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_split.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),  # Dropout layer for regularization
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),  # Dropout layer for regularization
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),  # Dropout in the fully connected layer
        Dense(len(y_train_split.unique()), activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare data for CNN
X_train_cnn = np.expand_dims(X_train_split, axis=2)
X_val_cnn = np.expand_dims(X_val_split, axis=2)

# Convert integer labels to categorical format
y_train_cnn = to_categorical(y_train_split)  # Use the integer-mapped y_train_split
y_val_cnn = to_categorical(y_val_split)  # Use the integer-mapped y_val_split

# Check the cardinality to avoid mismatches
print(f"X_train_cnn shape: {X_train_cnn.shape}, y_train_cnn shape: {y_train_cnn.shape}")
print(f"X_val_cnn shape: {X_val_cnn.shape}, y_val_cnn shape: {y_val_cnn.shape}")

# CNN Model training
cnn_model = create_cnn_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

cnn_history = cnn_model.fit(
    X_train_cnn, y_train_cnn, 
    epochs=20, 
    batch_size=32, 
    validation_data=(X_val_cnn, y_val_cnn), 
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate CNN on validation set
cnn_val_loss, cnn_val_accuracy = cnn_model.evaluate(X_val_cnn, y_val_cnn)
print(f"CNN Validation Accuracy: {cnn_val_accuracy}")

# Evaluate CNN on the test set
X_test_cnn = np.expand_dims(X_test_preprocessed, axis=2)
y_test_cnn = to_categorical(pd.Series(y_test_int).map(activity_name_to_index))
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(X_test_cnn, y_test_cnn)
print(f"CNN Test Accuracy: {cnn_test_accuracy}")
