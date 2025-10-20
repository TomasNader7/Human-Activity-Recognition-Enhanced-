"""
Stage 3: Cross-Dataset Evaluation with Model Training
Train on UCI HAR (source) → Test on WISDM (target)
Goal: generate confusion matrices and performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

# ============================================================================
# STEP 1: LOAD FILTERED DATASETS
# ============================================================================
def load_filtered_datasets(data_dir, dataset_type='3class'):

    print("=" * 80)
    print(f"LOADING FILTERED {dataset_type.upper()} DATASETS")
    print("=" * 80)

    # Load UCI HAR (source domain)
    uci_dir = os.path.join(data_dir, f'{dataset_type}_uci')
    X_train = pd.read_csv(os.path.join(uci_dir, 'X_filtered.txt'), sep=r'\s+', header=None)
    y_train = pd.read_csv(os.path.join(uci_dir, 'y_filtered.txt'), header=None, names=['activity'])

    # Load WISDM (target domain)
    wisdm_dir = os.path.join(data_dir, f'{dataset_type}_wisdm')
    X_test = pd.read_csv(os.path.join(wisdm_dir, 'X_filtered.txt'), sep=r'\s+', header=None)
    y_test = pd.read_csv(os.path.join(wisdm_dir, 'y_filtered.txt'), header=None, names=['activity'])

    # Load activity mapping
    activity_mapping = pd.read_csv(os.path.join(uci_dir, 'activity_mapping.csv'))
    activity_labels = dict(zip(activity_mapping['label'], activity_mapping['activity']))

    # Align features using feature selection on UCI HAR to match WISDM's feature count
    if X_train.shape[1] != X_test.shape[1]:
        print(f"Feature mismatch detected: UCI HAR has {X_train.shape[1]} features, WISDM has {X_test.shape[1]} features")
        target_features = X_test.shape[1]
        selector = SelectKBest(score_func=mutual_info_classif, k=target_features)
        X_train_selected = selector.fit_transform(X_train, y_train['activity'])
        # Convert back to DataFrame for consistency
        X_train = pd.DataFrame(X_train_selected)
        print(f"Selected top {target_features} features from UCI HAR using mutual information for compatibility.")
    
    print(f"  UCI HAR (Source - Training):")
    print(f"  Shape: {X_train.shape}")
    print(f"  Samples per activity:")
    for label, activity in sorted(activity_labels.items()):
        count = (y_train['activity'] == label).sum()
        print(f"    {label}. {activity}: {count:,}")
    
    print(f"\n  WISDM (Target - Testing):")
    print(f"  Shape: {X_test.shape}")
    print(f"  Samples per activity:")
    for label, activity in sorted(activity_labels.items()):
        count = (y_test['activity'] == label).sum()
        print(f"    {label}. {activity}: {count:,}")
    
    print()
    
    return X_train, y_train, X_test, y_test, activity_labels


# ===========================================================================================
# STEP 2: BUILD STACKING ENSEMBLE MODEL (From Research Paper - Human_Activity_Recognition.py)
# ===========================================================================================
def build_stacking_ensemble():
    print("=" * 80)
    print("BUILDING ADVANCED STACKING ENSEMBLE MODEL")
    print("=" * 80)

    # Base learners (Level 0)
    base_learners = [
        ('perceptron', Perceptron(max_iter=2000, tol=1e-3, random_state=42)),
        ('random_forest', RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
        ('xgboost', XGBClassifier(n_estimators=20, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='mlogloss')),
        ('gradient_boosting', GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42))
    ]

    # Meta-learner (Level 1)
    meta_learner = LogisticRegression(solver='saga', max_iter=2000, tol=1e-4, random_state=42)

    # Stacking ensemble
    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=3,  # 3-fold cross-validation
        n_jobs=-1
    )
    
    print("  Model Configuration:")
    print("  Base Learners (5):")
    for name, _ in base_learners:
        print(f"    - {name}")
    print("  Meta-Learner: Logistic Regression (SAGA solver)")
    print("  Cross-Validation: 3-fold")
    print()
    
    return stacking_model

# ============================================================================
# STEP 3: TRAIN MODEL ON SOURCE DOMAIN (UCI HAR)
# ============================================================================
def train_model(model, X_train, y_train):
    """Train the stacking ensemble on UCI HAR data"""
    print("=" * 80)
    print("TRAINING MODEL ON SOURCE DOMAIN (UCI HAR)")
    print("=" * 80)
    
    print("Training in progress...")
    print("  (This may take a few minutes with 3-fold CV on 5 base models)")
    
    # Train
    model.fit(X_train, y_train['activity'].values)
    
    # Evaluate on training data (source domain accuracy)
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train['activity'], y_pred_train)
    
    print(f"\n  Training Complete!")
    print(f"  Source Domain Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print()
    
    return model, train_accuracy

# ============================================================================
# STEP 4: EVALUATE ON TARGET DOMAIN (WISDM)
# ============================================================================
def evaluate_on_target(model, X_test, y_test, activity_labels, dataset_type, output_dir):
    """Evaluate trained model on WISDM and generate confusion matrix"""
    print("=" * 80)
    print("EVALUATING ON TARGET DOMAIN (WISDM)")
    print("=" * 80)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    test_accuracy = accuracy_score(y_test['activity'], y_pred)
    
    print(f"  Target Domain Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print()
    
    # Classification report
    activity_names = [activity_labels[label] for label in sorted(activity_labels.keys())]
    print("Classification Report:")
    print(classification_report(y_test['activity'], y_pred, target_names=activity_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test['activity'], y_pred)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, activity_names, test_accuracy, dataset_type, output_dir)
    
    return test_accuracy, cm

# ============================================================================
# STEP 5: PLOT CONFUSION MATRIX
# ============================================================================
def plot_confusion_matrix(cm, activity_names, accuracy, dataset_type, output_dir):
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activity_names, 
                yticklabels=activity_names,
                cbar_kws={'label': 'Number of Samples'},
                ax=ax)
    
    ax.set_title(f'Confusion Matrix - Cross-Dataset Evaluation ({dataset_type.upper()})\n'
                 f'UCI HAR (Train) → WISDM (Test) | Accuracy: {accuracy:.2%}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Activity', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Activity', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'confusion_matrix_{dataset_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved confusion matrix: {save_path}")
    plt.show()
    plt.close()

# ===================================================================================
# STEP 6: PERFORMANCE DEGRADATION ANALYSIS (Document and Visualize performance drop)
# ===================================================================================
def analyze_performance_degradation(train_acc, test_acc, dataset_type, output_dir):
    print("\n" + "=" * 80)
    print("PERFORMANCE DEGRADATION ANALYSIS")
    print("=" * 80)

    degradation = train_acc - test_acc
    degradation_pct = (degradation / train_acc) * 100
    
    print(f"Configuration: {dataset_type.upper()}")
    print(f"  Source Domain Accuracy (UCI HAR): {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Target Domain Accuracy (WISDM):   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Absolute Degradation:              {degradation:.4f} ({degradation*100:.2f}%)")
    print(f"  Relative Degradation:              {degradation_pct:.2f}% of source performance")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    accuracies = [train_acc, test_acc]
    labels = ['Source\n(UCI HAR)', 'Target\n(WISDM)']
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(labels, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add degradation arrow
    ax.annotate('', xy=(1, test_acc), xytext=(0, train_acc),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0.5, (train_acc + test_acc)/2, 
            f'Degradation:\n{degradation:.2%}\n({degradation_pct:.1f}%)',
            ha='center', va='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Degradation Analysis ({dataset_type.upper()})\n'
                 f'Stacking Ensemble Model',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(output_dir, f'performance_degradation_{dataset_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved degradation analysis: {save_path}")
    plt.show()
    plt.close()
    
    print("=" * 80 + "\n")
    
    return degradation, degradation_pct

# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================
def run_cross_dataset_evaluation(data_dir, dataset_type='3class', output_dir='results'):
    """
    Complete Stage 3 pipeline: Train on UCI HAR, Test on WISDM
    """

    print("\n")
    print("+" + "-" * 78 + "+")
    print("|" + " " * 20 + "STAGE 3: CROSS-DATASET EVALUATION" + " " * 25 + "|")
    print("|" + " " * 25 + f"Mode: {dataset_type.upper()}" + " " * (52 - len(dataset_type)) + "|")
    print("+" + "-" * 78 + "+")
    print("\n")

    # Step 1: Load data
    X_train, y_train, X_test, y_test, activity_labels = load_filtered_datasets(data_dir, dataset_type)
    
    # Step 2: Build model
    model = build_stacking_ensemble()
    
    # Step 3: Train on UCI HAR
    model, train_accuracy = train_model(model, X_train, y_train)
    
    # Step 4: Evaluate on WISDM
    result_dir = os.path.join(output_dir, f'stage3_{dataset_type}')
    test_accuracy, cm = evaluate_on_target(model, X_test, y_test, activity_labels, dataset_type, result_dir)

    # Step 5: Performance degradation analysis
    degradation, degradation_pct = analyze_performance_degradation(train_accuracy, test_accuracy, dataset_type, result_dir)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Configuration: {dataset_type.upper()}")
    print(f"Training samples (UCI HAR): {len(X_train):,}")
    print(f"Testing samples (WISDM): {len(X_test):,}")
    print(f"\nResults:")
    print(f"  Source accuracy: {train_accuracy:.2%}")
    print(f"  Target accuracy: {test_accuracy:.2%}")
    print(f"  Degradation: {degradation:.2%} ({degradation_pct:.1f}% relative)")
    print(f"\nGenerated files in {result_dir}/:")
    print(f"  - confusion_matrix_{dataset_type}.png")
    print(f"  - performance_degradation_{dataset_type}.png")
    print("=" * 80 + "\n")
    
    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'degradation': degradation,
        'confusion_matrix': cm,
        'activity_labels': activity_labels
    }


# ============================================================================
# STEP 7: COMPARATIVE ANALYSIS (3-Class vs 6-Class)
# ============================================================================
def compare_3class_vs_6class(results_3class, results_6class, output_dir):
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS: 3-CLASS vs 6-CLASS")
    print("=" * 80)

    # Create comparison table
    comparison = pd.DataFrame({
        'Configuration': ['3-Class', '6-Class'],
        'Source Accuracy': [results_3class['train_accuracy'], results_6class['train_accuracy']],
        'Target Accuracy': [results_3class['test_accuracy'], results_6class['test_accuracy']],
        'Degradation': [results_3class['degradation'], results_6class['degradation']]
    })
    
    print("\nPerformance Comparison:")
    print(comparison.to_string(index=False))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Accuracy comparison
    x = np.arange(2)
    width = 0.35
    
    axes[0].bar(x - width/2, 
                [results_3class['train_accuracy'], results_6class['train_accuracy']], 
                width, label='Source (UCI HAR)', color='#2ecc71', alpha=0.8)
    axes[0].bar(x + width/2, 
                [results_3class['test_accuracy'], results_6class['test_accuracy']], 
                width, label='Target (WISDM)', color='#e74c3c', alpha=0.8)
    
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['3-Class', '6-Class'])
    axes[0].legend()
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Right: Degradation comparison
    degradations = [results_3class['degradation'], results_6class['degradation']]
    bars = axes[1].bar(['3-Class', '6-Class'], degradations, 
                       color=['#3498db', '#9b59b6'], alpha=0.8, edgecolor='black')
    
    for bar, deg in zip(bars, degradations):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{deg:.2%}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    axes[1].set_ylabel('Performance Degradation', fontsize=12, fontweight='bold')
    axes[1].set_title('Domain Shift Impact', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(output_dir, 'comparison_3class_vs_6class.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved comparison: {save_path}")
    plt.show()
    plt.close()
    
    print("\nKey Insights:")
    if results_3class['test_accuracy'] > results_6class['test_accuracy']:
        print("  - 3-class shows higher target accuracy (perfect-match activities)")
    else:
        print("  - 6-class shows competitive accuracy despite approximate matches")
    
    print(f"   Degradation difference: {abs(results_3class['degradation'] - results_6class['degradation']):.2%}")
    print("=" * 80 + "\n")


# ============================================================================
# USAGE
# ============================================================================
if __name__ == "__main__":
    
    # UPDATE THIS PATH - where your Stage 2 filtered datasets are
    DATA_DIR = r"C:\Users\tomin\source\repos\Synthetic Image Detection\Filtered_datasets_and_KS_results"
    OUTPUT_DIR = r"C:\Users\tomin\source\repos\Synthetic Image Detection\Stage3_cross_val_Results"
    
    # ==== PRIMARY: 3-CLASS EVALUATION ====
    print("\n" + "= " * 40)
    print("PRIMARY ANALYSIS: 3-CLASS CROSS-DATASET EVALUATION")
    print("= " * 40 + "\n")
    
    results_3class = run_cross_dataset_evaluation(
        data_dir=DATA_DIR,
        dataset_type='3class',
        output_dir=OUTPUT_DIR
    )
    
    # ==== SECONDARY: 6-CLASS EVALUATION ====
    print("\n" + "= " * 40)
    print("SECONDARY ANALYSIS: 6-CLASS CROSS-DATASET EVALUATION")
    print("= " * 40 + "\n")
    
    results_6class = run_cross_dataset_evaluation(
        data_dir=DATA_DIR,
        dataset_type='6class',
        output_dir=OUTPUT_DIR
    )
    
    # ==== COMPARATIVE ANALYSIS ====
    compare_3class_vs_6class(results_3class, results_6class, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("  STAGE 3 COMPLETE - ALL RESULTS GENERATED")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - stage3_3class/confusion_matrix_3class.png")
    print("  - stage3_3class/performance_degradation_3class.png")
    print("  - stage3_6class/confusion_matrix_6class.png")
    print("  - stage3_6class/performance_degradation_6class.png")
    print("\n  - comparison_3class_vs_6class.png")
    print("=" * 80)