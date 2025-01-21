# -*- coding: utf-8 -*-

# Basic libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sci-kit
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

#%% Load data

df = pd.read_parquet('inadimplence.parquet')

#%% Descriptive statistics
for variable in df.columns:
    print(f'\n\nDescriptive statistics of {variable}:')
    print(df[variable].describe())
    
print(f'\n\nVariables: {df.columns}')

#%% Missing values

def report_missing(df):
    print(f'Número de linhas: {df.shape[0]} | Número de colunas: {df.shape[1]}')
    return pd.DataFrame({'Pct_missing': df.isna().mean().apply(lambda x: f"{x:.1%}"),
                          'Freq_missing': df.isna().sum().apply(lambda x: f"{x:,.0f}").replace(',','.')})

report_missing(df)

#%% Train and test datas

# X: explanatory variables, y: target
X = df.drop(columns='inadimplence')
y = df.inadimplence

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

#%% Model

# Create the model
tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Train the model
tree.fit(X_train, y_train)

#%% Evaluate function

def evaluate_tree(clf, y, X, y_labels=['0', '1'], data = 'train'):
    
    # Calcular as classificações preditas
    pred = clf.predict(X)
    
    # Calcular a probabilidade de evento
    y_prob = clf.predict_proba(X)[:, -1]
    
    # Calculando acurácia e matriz de confusão
    cm = confusion_matrix(y, pred)
    ac = accuracy_score(y, pred)
    bac = balanced_accuracy_score(y, pred)

    print(f'\ndata de {data}:')
    print(f'Tree accuracy: {ac:.1%}')
    print(f'Tree balanced accuracy: {bac:.1%}')
    
    # Calculando AUC
    auc_score = roc_auc_score(y, y_prob)
    print(f"AUC-ROC: {auc_score:.2%}")
    print(f"GINI: {(2*auc_score-1):.2%}")
    
    # Visualização gráfica
    sns.heatmap(cm, 
                annot=True, fmt='d', cmap='viridis', 
                xticklabels=y_labels, 
                yticklabels=y_labels)
    
    # Relatório de classificação do Scikit
    print('\n', classification_report(y, pred))
    
    # Gerar a Curva ROC
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    
    # Plotar a Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Linha de referência (modelo aleatório)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC - {data}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

#%% Evaluating the model

print('Evaluating the model on train data:')
evaluate_tree(tree, y_train, X_train, y_labels=['Adimplent', 'Inadimplent'], data='train')

print('Evaluating the model on test data:')
evaluate_tree(tree, y_test, X_test, y_labels=['Adimplent', 'Inadimplent'], data='test')