

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ============================================================
# SECTION 1: DATA LOADING & EXPLORATION
# ============================================================

print("=" * 60)
print("  DAY 3 — FLOWER SPECIES CLASSIFIER")
print("  Industrial Level ML Project")
print("=" * 60)

# Load Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
})

print("\n[1] DATASET INFO")
print("-" * 40)
print(f"Total Samples  : {len(df)}")
print(f"Total Features : {len(iris.feature_names)}")
print(f"Species        : {', '.join(iris.target_names)}")
print(f"\nFeatures:")
for f in iris.feature_names:
    print(f"  - {f}")

print(f"\nClass Distribution:")
print(df['species_name'].value_counts().to_string())

print(f"\nDataset Statistics:")
print(df.describe().round(2).to_string())

print(f"\nMissing Values: {df.isnull().sum().sum()} (None!)")


# ============================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n[2] GENERATING EDA PLOTS...")

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Flower Classifier — Exploratory Data Analysis', fontsize=18, fontweight='bold', y=0.98)

colors = {'Setosa': '#3B82F6', 'Versicolor': '#10B981', 'Virginica': '#F59E0B'}
species_colors = [colors[s] for s in df['species_name']]

# Plot 1: Species Distribution
ax1 = fig.add_subplot(3, 4, 1)
counts = df['species_name'].value_counts()
bars = ax1.bar(counts.index, counts.values, color=[colors[s] for s in counts.index], edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), ha='center', fontsize=11, fontweight='bold')
ax1.set_title('Species Distribution', fontweight='bold')
ax1.set_ylabel('Count')
ax1.set_ylim(0, 60)

# Plot 2-5: Feature distributions per species
features = iris.feature_names
for i, feature in enumerate(features):
    ax = fig.add_subplot(3, 4, i + 2)
    for species, color in colors.items():
        data = df[df['species_name'] == species][feature]
        ax.hist(data, bins=12, alpha=0.6, color=color, label=species, edgecolor='white')
    ax.set_title(feature.replace(' (cm)', ''), fontweight='bold')
    ax.set_xlabel('cm')
    ax.set_ylabel('Count')
    if i == 0:
        ax.legend(fontsize=8)

# Plot 6: Scatter — Sepal
ax6 = fig.add_subplot(3, 4, 6)
for species, color in colors.items():
    mask = df['species_name'] == species
    ax6.scatter(df[mask]['sepal length (cm)'], df[mask]['sepal width (cm)'],
                c=color, label=species, alpha=0.7, s=40, edgecolors='white', linewidth=0.3)
ax6.set_xlabel('Sepal Length (cm)')
ax6.set_ylabel('Sepal Width (cm)')
ax6.set_title('Sepal: Length vs Width', fontweight='bold')
ax6.legend(fontsize=8)

# Plot 7: Scatter — Petal
ax7 = fig.add_subplot(3, 4, 7)
for species, color in colors.items():
    mask = df['species_name'] == species
    ax7.scatter(df[mask]['petal length (cm)'], df[mask]['petal width (cm)'],
                c=color, label=species, alpha=0.7, s=40, edgecolors='white', linewidth=0.3)
ax7.set_xlabel('Petal Length (cm)')
ax7.set_ylabel('Petal Width (cm)')
ax7.set_title('Petal: Length vs Width', fontweight='bold')
ax7.legend(fontsize=8)

# Plot 8: Correlation Heatmap
ax8 = fig.add_subplot(3, 4, 8)
corr_matrix = df[iris.feature_names].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax8, cbar=True, square=True, linewidths=0.5)
ax8.set_title('Feature Correlation', fontweight='bold')
ax8.set_xticklabels([f.replace(' (cm)', '') for f in iris.feature_names], rotation=45, fontsize=8)
ax8.set_yticklabels([f.replace(' (cm)', '') for f in iris.feature_names], rotation=0, fontsize=8)

# Plot 9-12: Boxplots per feature
for i, feature in enumerate(features):
    ax = fig.add_subplot(3, 4, i + 9)
    data_by_species = [df[df['species_name'] == s][feature].values for s in colors.keys()]
    bp = ax.boxplot(data_by_species, patch_artist=True, labels=list(colors.keys()))
    for patch, color in zip(bp['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(feature.replace(' (cm)', ''), fontweight='bold')
    ax.set_ylabel('cm')
    ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("  EDA saved as: eda_analysis.png")


# ============================================================
# SECTION 3: DATA PREPROCESSING
# ============================================================

print("\n[3] DATA PREPROCESSING")
print("-" * 40)

X = df[iris.feature_names]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set : {X_train.shape[0]} samples")
print(f"Test set     : {X_test.shape[0]} samples")
print(f"Features     : {X_train.shape[1]}")
print(f"Scaler       : StandardScaler (mean=0, std=1)")


# ============================================================
# SECTION 4: MULTIPLE MODEL COMPARISON
# ============================================================

print("\n[4] TRAINING & COMPARING MULTIPLE MODELS")
print("-" * 40)

models = {
    'Decision Tree'       : DecisionTreeClassifier(random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM'                 : SVC(probability=True, random_state=42),
    'KNN'                 : KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    results[name] = {
        'model'    : model,
        'accuracy' : acc,
        'cv_mean'  : cv_scores.mean(),
        'cv_std'   : cv_scores.std(),
        'y_pred'   : y_pred
    }
    print(f"  {name:<22} Accuracy: {acc*100:.1f}%  CV: {cv_scores.mean()*100:.1f}% (+/-{cv_scores.std()*100:.1f}%)")


# ============================================================
# SECTION 5: HYPERPARAMETER TUNING (Random Forest)
# ============================================================

print("\n[5] HYPERPARAMETER TUNING — Random Forest")
print("-" * 40)

param_grid = {
    'n_estimators'  : [50, 100, 200],
    'max_depth'     : [None, 5, 10],
    'min_samples_split' : [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_scaled, y_train)

best_params  = grid_search.best_params_
best_score   = grid_search.best_score_
best_model   = grid_search.best_estimator_
best_pred    = best_model.predict(X_test_scaled)
best_acc     = accuracy_score(y_test, best_pred)

print(f"  Best Params  : {best_params}")
print(f"  CV Score     : {best_score*100:.2f}%")
print(f"  Test Accuracy: {best_acc*100:.2f}%")


# ============================================================
# SECTION 6: BEST MODEL EVALUATION
# ============================================================

print("\n[6] BEST MODEL — DETAILED EVALUATION")
print("-" * 40)

print(f"\nClassification Report:")
print(classification_report(
    y_test, best_pred,
    target_names=iris.target_names
))

feature_importance = pd.Series(
    best_model.feature_importances_,
    index=iris.feature_names
).sort_values(ascending=False)

print("Feature Importance:")
for feat, imp in feature_importance.items():
    bar = '█' * int(imp * 40)
    print(f"  {feat.replace(' (cm)',''):<20} {bar} {imp:.3f}")


# ============================================================
# SECTION 7: RESULTS VISUALIZATION
# ============================================================

print("\n[7] GENERATING RESULTS PLOTS...")

fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
fig2.suptitle('Flower Classifier — Model Results', fontsize=16, fontweight='bold')

# Plot 1: Model Comparison
ax = axes[0][0]
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] * 100 for m in model_names]
short_names = ['DT', 'RF', 'GB', 'SVM', 'KNN', 'LR']
bar_colors = ['#3B82F6' if acc < max(accuracies) else '#10B981' for acc in accuracies]
bars = ax.bar(short_names, accuracies, color=bar_colors, edgecolor='black', linewidth=0.8)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(80, 105)
ax.set_title('Model Accuracy Comparison', fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Model')

# Plot 2: CV Score comparison
ax = axes[0][1]
cv_means = [results[m]['cv_mean'] * 100 for m in model_names]
cv_stds  = [results[m]['cv_std'] * 100 for m in model_names]
ax.barh(short_names, cv_means, xerr=cv_stds, color='#8B5CF6',
        edgecolor='black', linewidth=0.8, capsize=4, alpha=0.8)
ax.set_title('5-Fold Cross Validation', fontweight='bold')
ax.set_xlabel('CV Accuracy (%)')
ax.set_xlim(80, 105)

# Plot 3: Confusion Matrix (Best Model)
ax = axes[0][2]
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            linewidths=0.5)
ax.set_title(f'Confusion Matrix\n(Random Forest - Best Model)', fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# Plot 4: Feature Importance
ax = axes[1][0]
feat_names = [f.replace(' (cm)', '') for f in feature_importance.index]
colors_feat = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444']
bars = ax.barh(feat_names, feature_importance.values, color=colors_feat, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, feature_importance.values):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)
ax.set_title('Feature Importance', fontweight='bold')
ax.set_xlabel('Importance Score')
ax.set_xlim(0, 0.6)
ax.invert_yaxis()

# Plot 5: Decision Boundary (Petal features)
ax = axes[1][1]
X_petal = X_test_scaled[:, 2:]
h = 0.02
x_min, x_max = X_petal[:, 0].min() - 1, X_petal[:, 0].max() + 1
y_min, y_max = X_petal[:, 1].min() - 1, X_petal[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

dummy_X = np.zeros((xx.ravel().shape[0], 4))
dummy_X[:, 2] = xx.ravel()
dummy_X[:, 3] = yy.ravel()
Z = best_model.predict(dummy_X).reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
scatter_colors_test = ['#3B82F6' if c == 0 else '#10B981' if c == 1 else '#F59E0B' for c in y_test]
ax.scatter(X_petal[:, 0], X_petal[:, 1], c=scatter_colors_test, s=40, edgecolors='black', linewidth=0.5)
ax.set_title('Decision Boundary\n(Petal Features)', fontweight='bold')
ax.set_xlabel('Petal Length (scaled)')
ax.set_ylabel('Petal Width (scaled)')

# Plot 6: Prediction Summary
ax = axes[1][2]
correct = (best_pred == y_test.values).sum()
wrong   = len(y_test) - correct
wedge_colors = ['#10B981', '#EF4444']
wedges, texts, autotexts = ax.pie(
    [correct, wrong],
    labels=[f'Correct\n({correct})', f'Wrong\n({wrong})'],
    colors=wedge_colors,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 11}
)
ax.set_title(f'Prediction Summary\nTest Set ({len(y_test)} samples)', fontweight='bold')

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Results saved as: model_results.png")


# ============================================================
# SECTION 8: SAVE MODEL
# ============================================================

print("\n[8] SAVING MODEL...")
print("-" * 40)

model_data = {
    'model'          : best_model,
    'scaler'         : scaler,
    'feature_names'  : iris.feature_names,
    'target_names'   : iris.target_names,
    'best_params'    : best_params,
    'accuracy'       : best_acc
}

with open('flower_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("  Model saved as: flower_model.pkl")
print(f"  Final Accuracy : {best_acc*100:.2f}%")


# ============================================================
# SECTION 9: GUI APP
# ============================================================

print("\n[9] LAUNCHING GUI APP...")

def predict_flower():
    try:
        sl = float(sepal_length.get())
        sw = float(sepal_width.get())
        pl = float(petal_length.get())
        pw = float(petal_width.get())

        if not (0 < sl < 20 and 0 < sw < 20 and 0 < pl < 20 and 0 < pw < 20):
            messagebox.showwarning("Warning", "Values 0-20 cm ke beech hone chahiye!")
            return

        features = np.array([[sl, sw, pl, pw]])
        features_scaled = scaler.transform(features)
        prediction = best_model.predict(features_scaled)[0]
        probabilities = best_model.predict_proba(features_scaled)[0]
        species_name = iris.target_names[prediction]
        confidence = probabilities[prediction] * 100

        emoji_map = {'setosa': '🌸', 'versicolor': '🌺', 'virginica': '🌼'}
        color_map  = {'setosa': '#3B82F6', 'versicolor': '#10B981', 'virginica': '#F59E0B'}

        result_var.set(f"{emoji_map[species_name]}  {species_name.upper()}")
        conf_var.set(f"Confidence: {confidence:.1f}%")
        result_label.config(fg=color_map[species_name])

        prob_text = "  |  ".join([
            f"{iris.target_names[i].capitalize()}: {probabilities[i]*100:.1f}%"
            for i in range(3)
        ])
        prob_var.set(prob_text)

    except ValueError:
        messagebox.showerror("Error", "Sirf numbers daalo!")

def reset_fields():
    for var in [sl_var, sw_var, pl_var, pw_var]:
        var.set("")
    result_var.set("")
    conf_var.set("")
    prob_var.set("")

root = tk.Tk()
root.title("Flower Species Classifier")
root.geometry("520x560")
root.configure(bg='#0F172A')
root.resizable(False, False)

title_frame = tk.Frame(root, bg='#1E293B', pady=12)
title_frame.pack(fill='x')
tk.Label(title_frame, text="🌸 Flower Species Classifier",
         font=('Arial', 18, 'bold'), bg='#1E293B', fg='white').pack()
tk.Label(title_frame, text=f"Random Forest  |  Accuracy: {best_acc*100:.1f}%",
         font=('Arial', 10), bg='#1E293B', fg='#94A3B8').pack()

input_frame = tk.Frame(root, bg='#0F172A', pady=10)
input_frame.pack(fill='x', padx=20)

fields = [
    ('Sepal Length (cm)', 'e.g. 5.1'),
    ('Sepal Width  (cm)', 'e.g. 3.5'),
    ('Petal Length (cm)', 'e.g. 1.4'),
    ('Petal Width  (cm)', 'e.g. 0.2')
]

sl_var = tk.StringVar(); sw_var = tk.StringVar()
pl_var = tk.StringVar(); pw_var = tk.StringVar()
vars_list = [sl_var, sw_var, pl_var, pw_var]
entries = []

for i, (label, placeholder) in enumerate(fields):
    row = tk.Frame(input_frame, bg='#0F172A')
    row.pack(fill='x', pady=5)
    tk.Label(row, text=label, font=('Arial', 11), bg='#0F172A',
             fg='#CBD5E1', width=20, anchor='w').pack(side='left')
    entry = tk.Entry(row, textvariable=vars_list[i], font=('Arial', 11),
                     bg='#1E293B', fg='white', insertbackground='white',
                     relief='flat', bd=0, highlightthickness=1,
                     highlightbackground='#334155', highlightcolor='#3B82F6',
                     width=15)
    entry.pack(side='left', ipady=6, padx=5)
    tk.Label(row, text=placeholder, font=('Arial', 9), bg='#0F172A',
             fg='#475569').pack(side='left')
    entries.append(entry)

sepal_length, sepal_width, petal_length, petal_width = entries

btn_frame = tk.Frame(root, bg='#0F172A')
btn_frame.pack(pady=15)

tk.Button(btn_frame, text="Classify Flower",
          font=('Arial', 13, 'bold'), bg='#3B82F6', fg='white',
          relief='flat', padx=25, pady=10, cursor='hand2',
          command=predict_flower).pack(side='left', padx=8)

tk.Button(btn_frame, text="Reset",
          font=('Arial', 12), bg='#334155', fg='white',
          relief='flat', padx=20, pady=10, cursor='hand2',
          command=reset_fields).pack(side='left', padx=8)

result_frame = tk.Frame(root, bg='#1E293B', pady=15, padx=20)
result_frame.pack(fill='x', padx=20)

result_var = tk.StringVar()
conf_var   = tk.StringVar()
prob_var   = tk.StringVar()

result_label = tk.Label(result_frame, textvariable=result_var,
                        font=('Arial', 22, 'bold'), bg='#1E293B', fg='white')
result_label.pack()

tk.Label(result_frame, textvariable=conf_var,
         font=('Arial', 12), bg='#1E293B', fg='#94A3B8').pack(pady=4)

tk.Label(result_frame, textvariable=prob_var,
         font=('Arial', 9), bg='#1E293B', fg='#64748B',
         wraplength=460).pack()

sample_frame = tk.Frame(root, bg='#0F172A', pady=5)
sample_frame.pack(fill='x', padx=20)
tk.Label(sample_frame, text="Sample Test Values:",
         font=('Arial', 10, 'bold'), bg='#0F172A', fg='#94A3B8').pack(anchor='w')

samples = [
    ("Setosa",     "5.1, 3.5, 1.4, 0.2"),
    ("Versicolor", "6.0, 2.7, 4.1, 1.0"),
    ("Virginica",  "6.3, 3.3, 6.0, 2.5")
]
sample_colors = {'Setosa': '#3B82F6', 'Versicolor': '#10B981', 'Virginica': '#F59E0B'}
for species, values in samples:
    row = tk.Frame(sample_frame, bg='#0F172A')
    row.pack(anchor='w')
    tk.Label(row, text=f"  {species}:", font=('Arial', 10, 'bold'),
             bg='#0F172A', fg=sample_colors[species], width=12, anchor='w').pack(side='left')
    tk.Label(row, text=values, font=('Arial', 10),
             bg='#0F172A', fg='#64748B').pack(side='left')

tk.Label(root, text=f"Model: Random Forest  |  Test Accuracy: {best_acc*100:.1f}%  |  Day 3 of 7",
         font=('Arial', 9), bg='#0F172A', fg='#334155').pack(side='bottom', pady=8)

print("  GUI App launched!")
root.mainloop()

print("\n" + "=" * 60)
print("  DAY 3 COMPLETE!")
print(f"  Best Model  : Random Forest")
print(f"  Accuracy    : {best_acc*100:.2f}%")
print(f"  Files saved : eda_analysis.png, model_results.png, flower_model.pkl")
print("=" * 60)
