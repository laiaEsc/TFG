# -*- coding: utf-8 -*-

#################################################################################
###     CONFIGURACIÓ
#################################################################################
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dbfread import DBF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
from sklearn.utils import resample
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
import tensorflow as tf
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from xgboost import DMatrix, train
from sklearn.metrics import average_precision_score

## configuració consola
pd.set_option('display.max_columns', None)  # Mostra totes les columnes
pd.set_option('display.max_rows', None)     # Mostra totes les files
pd.set_option('display.width', 1000)        # Amplia l'amplada per evitar truncaments
sns.set(style="whitegrid")


#################################################################################
###     CONFIGURACIÓ PER LA IMPORTACIÓ DE FITXERS
#################################################################################
df_fire_sprd_path = 'C:\\Users\\laia\\OneDrive\\MatCAD\\4t.2\\TFG\\nfirs_all_incident_pdr_2023\\nfirs_all_incident_pdr_2023\\df_fire_sprd.csv'

# Importar fitxers
df_fire_sprd = pd.read_csv(df_fire_sprd_path, delimiter=',', encoding='unicode_escape')

# Primera visualització de les dades
print(df_fire_sprd.head())
print(df_fire_sprd.info())
print(df_fire_sprd.describe())
print(df_fire_sprd.isnull().sum())


#################################################################################
###     ENTRENAMENT DE MODELS
#################################################################################
# Seperació de les dades
X = df_fire_sprd.drop(columns=['FIRE_SPRD', 'INCIDENT_KEY'])  # Característiques
y = df_fire_sprd['FIRE_SPRD']  # Variable objectiu

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

############################    RANDOM FOREST    ###############################
# Model inicial
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1-Score:", f1_score(y_test, y_pred_rf, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió de RandomForest")
plt.show()

# Ajustant hiperparàmetres
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced', random_state=42),
                           param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

print("Millors hiperparàmetres trobats:", grid_search_rf.best_params_) # Millors hiperparàmetres trobats: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 300}

best_rf = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print("F1-Score:", f1_score(y_test, y_pred_best_rf, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_best_rf))

cm = confusion_matrix(y_test, y_pred_best_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió de RandomForest amb GridSearch")
plt.show()

# Selecció de característiques
importances = rf.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)

## seleccionem les que tenen més importància
top_features = feature_importance_df.head(31)['Feature']
X_train_top31 = X_train[top_features]
X_test_top31 = X_test[top_features]

rf_top31 = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_top31.fit(X_train_top31, y_train)
y_pred_top_rf31 = rf_top31.predict(X_test_top31)

print("Accuracy:", accuracy_score(y_test, y_pred_top_rf31))
print("F1-Score:", f1_score(y_test, y_pred_top_rf31, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_top_rf31))

cm = confusion_matrix(y_test, y_pred_top_rf31)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió de RandomForest amb 31 paràmetres")
plt.show()

## sense reduir tant
top_features = feature_importance_df.head(56)['Feature']
X_train_top56 = X_train[top_features]
X_test_top56 = X_test[top_features]

rf_top56 = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_top56.fit(X_train_top56, y_train)
y_pred_top_rf56 = rf_top56.predict(X_test_top56)

print("Accuracy:", accuracy_score(y_test, y_pred_top_rf56))
print("F1-Score:", f1_score(y_test, y_pred_top_rf56, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_top_rf56))

cm = confusion_matrix(y_test, y_pred_top_rf56)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió de RandomForest amb 56 paràmetres")
plt.show()


######################    GRADIENT BOOSTING: XGBoost    ########################
xgb = XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("F1-Score:", f1_score(y_test, y_pred_xgb, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_xgb))

cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió del XGBoost")
plt.show()

# Ajustant hiperparàmetres
## 1r model
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search_xgb = GridSearchCV(estimator=XGBClassifier(objective='multi:softmax', num_class=3, random_state=42),
                           param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
grid_search_xgb.fit(X_train, y_train)

print("Millors hiperparàmetres trobats:", grid_search_xgb.best_params_) # Millors hiperparàmetres trobats: Millors hiperparàmetres trobats: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 5, 'n_estimators': 200, 'subsample': 0.8}

best_xgb = grid_search_xgb.best_estimator_
y_pred_best_xgb = best_xgb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_best_xgb))
print("F1-Score:", f1_score(y_test, y_pred_best_xgb, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_best_xgb))

cm = confusion_matrix(y_test, y_pred_best_xgb)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió del XGBoost amb GridSearch")
plt.show()

## 2n model
param_dist = {
    'n_estimators': np.arange(100, 301, 50),
    'learning_rate': np.linspace(0.01, 0.2, 10),
    'max_depth': np.arange(3, 10),
    'min_child_weight': np.arange(1, 6),
    'subsample': np.linspace(0.7, 1.0, 5),
    'colsample_bytree': np.linspace(0.7, 1.0, 5),
    'gamma': np.linspace(0, 0.5, 5),
    'reg_alpha': np.linspace(0, 1, 5),
    'reg_lambda': np.linspace(1, 10, 5)
}

random_search_xgb = RandomizedSearchCV(estimator=XGBClassifier(objective='multi:softmax', num_class=3, random_state=42),
                                       param_distributions=param_dist, n_iter=100, scoring='f1_weighted', cv=3,
                                       random_state=42, verbose=1, n_jobs=-1)
random_search_xgb.fit(X_train, y_train)

print("Millors hiperparàmetres trobats:", random_search_xgb.best_params_) #Millors hiperparàmetres trobats: {'subsample': 0.925, 'reg_lambda': 5.5, 'reg_alpha': 0.25, 'n_estimators': 250, 'min_child_weight': 4, 'max_depth': 5, 'learning_rate': 0.11555555555555555, 'gamma': 0.0, 'colsample_bytree': 0.85}

best_xgb2 = random_search_xgb.best_estimator_
y_pred_best_xgb2 = best_xgb2.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_best_xgb2))
print("F1-Score:", f1_score(y_test, y_pred_best_xgb2, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_best_xgb2))

cm = confusion_matrix(y_test, y_pred_best_xgb2)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió del XGBoost amb RandomizedSearchCV")
plt.show()

## 3r model
param_dist = {
    'n_estimators': np.arange(200, 301, 50),
    'learning_rate': np.linspace(0.1, 0.2, 10),
    'max_depth': np.arange(3, 8),
    'min_child_weight': np.arange(3, 5),
    'subsample': np.linspace(0.9, 1.0, 5),
    'colsample_bytree': np.linspace(0.8, 0.9, 5),
    'gamma': np.linspace(0, 0.2, 5),
    'reg_alpha': np.linspace(0, 0.25, 5),
    'reg_lambda': np.linspace(3, 7, 5)
}

random_search_xgb3 = RandomizedSearchCV(estimator=XGBClassifier(objective='multi:softmax', num_class=3, random_state=42),
                                       param_distributions=param_dist, n_iter=100, scoring='f1_weighted', cv=3,
                                       random_state=42, verbose=1, n_jobs=-1)
random_search_xgb3.fit(X_train, y_train)

print("Millors hiperparàmetres trobats:", random_search_xgb3.best_params_) #Millors hiperparàmetres trobats: {'subsample': 0.925, 'reg_lambda': 4.0, 'reg_alpha': 0.0, 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 5, 'learning_rate': 0.13333333333333333, 'gamma': 0.2, 'colsample_bytree': 0.9}

best_xgb3 = random_search_xgb3.best_estimator_
y_pred_best_xgb3 = best_xgb3.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_best_xgb3))
print("F1-Score:", f1_score(y_test, y_pred_best_xgb3, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_best_xgb3))

cm = confusion_matrix(y_test, y_pred_best_xgb3)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió del XGBoost amb RandomizedSearchCV2")
plt.show()


######################    GRADIENT BOOSTING: LightGBM    ########################
lgb = LGBMClassifier(class_weight='balanced', random_state=42)
lgb.fit(X_train, y_train)

y_pred_lgb = lgb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("F1-Score:", f1_score(y_test, y_pred_lgb, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_lgb))

cm = confusion_matrix(y_test, y_pred_lgb)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió de LightGBM")
plt.show()

# Ajustant hiperparàmetres
param_dist = {
    'num_leaves': np.arange(15, 128, 31),
    'max_depth': np.arange(3, 10, 2),
    'learning_rate': np.linspace(0.01, 0.2, 5),
    'n_estimators': [100, 200, 300, 400],
    'min_child_samples': [10, 20, 30],
    'subsample': np.linspace(0.7, 1.0, 5),
    'colsample_bytree': np.linspace(0.7, 1.0, 5),
    'reg_alpha': np.linspace(0, 1, 5),
    'reg_lambda': [1, 5, 10]
}

random_search_lgb = RandomizedSearchCV(estimator=LGBMClassifier(class_weight='balanced', random_state=42),
                                       param_distributions=param_dist, n_iter=100, scoring='f1_weighted', cv=3,
                                       random_state=42, n_jobs=-1, verbose=1)
random_search_lgb.fit(X_train, y_train)

print("Millors hiperparàmetres:", random_search_lgb.best_params_) #Millors hiperparàmetres: {'subsample': 0.925, 'reg_lambda': 10, 'reg_alpha': 0.5, 'num_leaves': 15, 'n_estimators': 300, 'min_child_samples': 20, 'max_depth': 7, 'learning_rate': 0.15250000000000002, 'colsample_bytree': 0.7749999999999999}

best_lgb = random_search_xgb3.best_estimator_
y_pred_best_lgb = best_lgb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_best_lgb))
print("F1-Score:", f1_score(y_test, y_pred_best_lgb, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_best_lgb))

cm = confusion_matrix(y_test, y_pred_best_lgb)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió del LightGBM amb RandomizedSearchCV")
plt.show()


######################    GRADIENT BOOSTING: CatBoost    ########################

cb = CatBoostClassifier(objective='MultiClass', random_state=42, verbose=100)
cb.fit(X_train, y_train)
y_pred_cd = cb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_cd))
print("F1-Score:", f1_score(y_test, y_pred_cd, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_cd))

cm = confusion_matrix(y_test, y_pred_cd)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió de CatBoost")
plt.show()

# Ajustant hiperparàmetres
param_dist = {
    'iterations': [100, 300, 500, 700, 1000],
    'learning_rate': np.linspace(0.01, 0.2, 5),
    'depth': [3, 5, 7, 9, 11],
    'l2_leaf_reg': [1, 3, 5, 7, 10],
    'bagging_temperature': [0, 0.1, 0.5, 1, 2],
    'border_count': [32, 64, 128, 256],
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
}

random_search_cb = RandomizedSearchCV(estimator=CatBoostClassifier(objective='MultiClass', random_state=42, verbose=0),
                                      param_distributions=param_dist, n_iter=75, scoring='f1_weighted', cv=3,
                                      random_state=42, n_jobs=-1, verbose=1)
random_search_cb.fit(X_train, y_train)

print("Millors hiperparàmetres:", random_search_cb.best_params_) #Millors hiperparàmetres: {'learning_rate': 0.15250000000000002, 'l2_leaf_reg': 1, 'iterations': 300, 'grow_policy': 'SymmetricTree', 'depth': 5, 'border_count': 128, 'bagging_temperature': 0.1}

best_cb = random_search_cb.best_estimator_
y_pred_best_cb = best_cb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_best_cb))
print("F1-Score:", f1_score(y_test, y_pred_best_cb, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_best_cb))

cm = confusion_matrix(y_test, y_pred_best_cb)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió del CatBoost amb RandomizedSearchCV")
plt.show()


######################    STACKING MODELS    ########################
# Model amb RF i XGB
estimators = [
    ('xgb', XGBClassifier(**random_search_xgb.best_params_)),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
]

stacking_model = StackingClassifier(estimators=estimators, final_estimator=XGBClassifier(random_state=42))
stacking_model.fit(X_train, y_train)
y_pred_stacking = stacking_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_stacking))
print("F1-Score:", f1_score(y_test, y_pred_stacking, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_stacking))

cm = confusion_matrix(y_test, y_pred_stacking)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió de STACKING")
plt.show()

# Model amb XGB i LGB
estimators = [
        ('xgb', XGBClassifier(random_state=42)),
        ('lgb', LGBMClassifier(random_state=42))
]

stacking_model2 = StackingClassifier(estimators=estimators, final_estimator=LGBMClassifier(random_state=42))
stacking_model2.fit(X_train, y_train)
y_pred_stacking2 = stacking_model2.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_stacking2))
print("F1-Score:", f1_score(y_test, y_pred_stacking2, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_stacking2))

cm = confusion_matrix(y_test, y_pred_stacking2)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió de STACKING2")
plt.show()

###########################    XARXES NEURONALS    #############################
# XN 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xn = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),  # Capa d'entrada amb 64 neurones
    Dense(32, activation='relu'),                              # Capa intermèdia amb 32 neurones
    Dense(3, activation='softmax')                             # Capa de sortida per a classificació multiclasse
])

xn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = xn.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

y_pred_proba = xn.predict(X_test_scaled)
y_pred_xn = np.argmax(y_pred_proba, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred_xn))
print("F1-Score:", f1_score(y_test, y_pred_xn, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_xn))

cm = confusion_matrix(y_test, y_pred_xn)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió de la Xarxa Neuronal")
plt.show()

# XN 2
xn2 = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(3, activation='softmax')
])

xn2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[keras.metrics.F1Score(average=None, threshold=None, name="f1_score", dtype=None)])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = xn2.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

y_pred_proba2 = xn2.predict(X_test_scaled)
y_pred_xn2 = np.argmax(y_pred_proba2, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred_xn2))
print("F1-Score:", f1_score(y_test, y_pred_xn2, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_xn2))

cm = confusion_matrix(y_test, y_pred_xn2)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió de la Xarxa Neuronal")
plt.show()

####################    FUNCIO OBJECTIU PERSONALITZADA    ####################### (per minimitzar infraestimació)
# Definim la Funció Objectiu Personalitzada per la Matriu de Cost
def custom_objective(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred = y_pred.reshape(-1, 3)

    labels_one_hot = np.eye(3)[y_true.astype(int)] # Fer One-Hot de les etiquetes

    # Calcular la Penalització segons Matriu de Cost
    grad = y_pred - labels_one_hot
    for i in range(len(y_true)):
        grad[i] = grad[i] * cost_matrix[int(y_true[i])]

    hess = np.ones_like(grad)

    return grad.ravel(), hess.ravel()

# Definim Mètrica Personalizada per la Matriu de Cost
def cost_based_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred_classes = np.argmax(y_pred, axis=1)

    total_cost = 0
    for true_class, pred_class in zip(y_true, y_pred_classes):
        total_cost += cost_matrix[int(true_class), int(pred_class)]

    avg_cost = total_cost / len(y_true)
    return 'avg_cost', avg_cost

# Definim Matriu de Cost
cost_matrix = np.array([
    [0, 2, 2],
    [4, 0, 2],
    [4, 4, 0]
])

# Convertir dades a DMatrix
dtrain = DMatrix(X_train, label=y_train)
dtest = DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'max_depth': 6,
    'eta': 0.1,
    'seed': 42
}

model_fop = train(params, dtrain, num_boost_round=100, obj=custom_objective, evals=[(dtest, 'test')],
              feval=cost_based_metric)

y_pred_probs = model_fop.predict(dtest)
y_pred_fop = np.argmax(y_pred_probs, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred_fop))
print("F1-Score:", f1_score(y_test, y_pred_fop, average='weighted'))
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred_fop))

cm = confusion_matrix(y_test, y_pred_fop)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Prediccions")
plt.ylabel("Realitat")
plt.title("Matriu de Confusió de la Funció Objectiu Personalitzada")
plt.show()


#################################################################################
###     DEFINICIÓ MÈTRICA PERSONALITZADA (per minimitzar infraestimació)
#################################################################################
cost_matrix = np.array([
    [0, 1, 1],
    [2, 0, 1],
    [2, 2, 0]
])

def metric_sobreestimacio(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    total_cost = 0
    for i in range(len(cm)):
        for j in range(len(cm)):
            total_cost += cm[i, j] * cost_matrix[i, j]

    avg_cost = total_cost / np.sum(cm)
    return avg_cost


#################################################################################
###     COMPARACIÓ DE MODELS
#################################################################################
# Cross Validation amb Accuracy
accuracy_cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
accuracy_cv_scores_best_rf = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='accuracy')
accuracy_cv_scores_rf_top31 = cross_val_score(rf_top31, X_train_top31, y_train, cv=5, scoring='accuracy')
accuracy_cv_scores_rf_top56 = cross_val_score(rf_top56, X_train_top56, y_train, cv=5, scoring='accuracy')

accuracy_cv_scores_xgb = cross_val_score(xgb, X_train, y_train, cv=5, scoring='accuracy')
accuracy_cv_scores_best_xgb = cross_val_score(best_xgb, X_train, y_train, cv=5, scoring='accuracy')
accuracy_cv_scores_best_xgb2 = cross_val_score(best_xgb2, X_train, y_train, cv=5, scoring='accuracy')
accuracy_cv_scores_best_xgb3 = cross_val_score(best_xgb3, X_train, y_train, cv=5, scoring='accuracy')

accuracy_cv_scores_lgb = cross_val_score(lgb, X_train, y_train, cv=5, scoring='accuracy')
accuracy_cv_scores_best_lgb = cross_val_score(best_lgb, X_train, y_train, cv=5, scoring='accuracy')

accuracy_cv_scores_cat = cross_val_score(cb, X_train, y_train, cv=5, scoring='accuracy')
accuracy_cv_scores_best_cat = cross_val_score(best_cb, X_train, y_train, cv=5, scoring='accuracy')

accuracy_cv_scores_stk = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring='accuracy')
accuracy_cv_scores_stk2 = cross_val_score(stacking_model2, X_train, y_train, cv=5, scoring='accuracy')

accuracy_cv_scores_xn = cross_val_score(xn, X_train, y_train, cv=5, scoring='accuracy')
accuracy_cv_scores_xn2 = cross_val_score(xn2, X_train, y_train, cv=5, scoring='accuracy')

accuracy_fop = accuracy_score(y_test, y_pred_fop)

print("Models de RandomForest:")
print("Accuracy mitjà (5-fold CV) per rf:", accuracy_cv_scores_rf.mean())
print("Accuracy mitjà (5-fold CV) per best_rf:", accuracy_cv_scores_best_rf.mean())
print("Accuracy mitjà (5-fold CV) per rf_top31:", accuracy_cv_scores_rf_top31.mean())
print("Accuracy mitjà (5-fold CV) per rf_top56:", accuracy_cv_scores_rf_top56.mean())
print("Models de XGBoost")
print("Accuracy mitjà (5-fold CV) per xgb:", accuracy_cv_scores_xgb.mean())
print("Accuracy mitjà (5-fold CV) per best_xgb:", accuracy_cv_scores_best_xgb.mean())
print("Accuracy mitjà (5-fold CV) per best_xgb2:", accuracy_cv_scores_best_xgb2.mean())
print("Accuracy mitjà (5-fold CV) per best_xgb3:", accuracy_cv_scores_best_xgb3.mean())
print("Models de LGB")
print("Accuracy mitjà (5-fold CV) per lgb:", accuracy_cv_scores_lgb.mean())
print("Accuracy mitjà (5-fold CV) per best_lgb:", accuracy_cv_scores_best_lgb.mean())
print("Models de CatBoost")
print("Accuracy mitjà (5-fold CV) per model_cat:", accuracy_cv_scores_cat.mean())
print("Accuracy mitjà (5-fold CV) per best_cat:", accuracy_cv_scores_best_cat.mean())
print("Models de Stacking")
print("Accuracy mitjà (5-fold CV) per stacking_model1:", accuracy_cv_scores_stk.mean())
print("Accuracy mitjà (5-fold CV) per stacking_model2:", accuracy_cv_scores_stk2.mean())
print("Xarxes Neuronals:")
print("Accuracy mitjà (5-fold CV) per xn1:", accuracy_cv_scores_xn.mean())
print("Accuracy mitjà (5-fold CV) per xn2:", accuracy_cv_scores_xn2.mean())
print("Models de FOP")
print("Accuracy per FOP:", accuracy_fop)

# Cross Validation amb Precision
precision_cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='average_precision')
precision_cv_scores_best_rf = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='average_precision')
precision_cv_scores_rf_top31 = cross_val_score(rf_top31, X_train_top31, y_train, cv=5, scoring='average_precision')
precision_cv_scores_rf_top56 = cross_val_score(rf_top56, X_train_top56, y_train, cv=5, scoring='average_precision')

precision_cv_scores_xgb = cross_val_score(xgb, X_train, y_train, cv=5, scoring='average_precision')
precision_cv_scores_best_xgb = cross_val_score(best_xgb, X_train, y_train, cv=5, scoring='average_precision')
precision_cv_scores_best_xgb2 = cross_val_score(best_xgb2, X_train, y_train, cv=5, scoring='average_precision')
precision_cv_scores_best_xgb3 = cross_val_score(best_xgb3, X_train, y_train, cv=5, scoring='average_precision')

precision_cv_scores_lgb = cross_val_score(lgb, X_train, y_train, cv=5, scoring='average_precision')
precision_cv_scores_best_lgb = cross_val_score(best_lgb, X_train, y_train, cv=5, scoring='average_precision')

precision_cv_scores_cat = cross_val_score(cb, X_train, y_train, cv=5, scoring='average_precision')
precision_cv_scores_best_cat = cross_val_score(best_cb, X_train, y_train, cv=5, scoring='average_precision')

precision_cv_scores_stk = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring='average_precision')
precision_cv_scores_stk2 = cross_val_score(stacking_model2, X_train, y_train, cv=5, scoring='average_precision')

precision_cv_scores_xn = cross_val_score(xn, X_train, y_train, cv=5, scoring='average_precision')
precision_cv_scores_xn2 = cross_val_score(xn2, X_train, y_train, cv=5, scoring='average_precision')

precision_fop = average_precision_score(y_test, y_pred_proba2, average='weighted')

print("Models de RandomForest:")
print("Precision mitjà (5-fold CV) per rf:", precision_cv_scores_rf.mean())
print("Precision mitjà (5-fold CV) per best_rf:", precision_cv_scores_best_rf.mean())
print("Precision mitjà (5-fold CV) per rf_top31:", precision_cv_scores_rf_top31.mean())
print("Precision mitjà (5-fold CV) per rf_top56:", precision_cv_scores_rf_top56.mean())
print("Models de XGBoost")
print("Precision mitjà (5-fold CV) per xgb:", precision_cv_scores_xgb.mean())
print("Precision mitjà (5-fold CV) per best_xgb:", precision_cv_scores_best_xgb.mean())
print("Precision mitjà (5-fold CV) per best_xgb2:", precision_cv_scores_best_xgb2.mean())
print("Precision mitjà (5-fold CV) per best_xgb3:", precision_cv_scores_best_xgb3.mean())
print("Models de LGB")
print("Precision mitjà (5-fold CV) per lgb:", precision_cv_scores_lgb.mean())
print("Precision mitjà (5-fold CV) per best_lgb:", precision_cv_scores_best_lgb.mean())
print("Models de CatBoost")
print("Precision mitjà (5-fold CV) per model_cat:", precision_cv_scores_cat.mean())
print("Precision mitjà (5-fold CV) per best_cat:", precision_cv_scores_best_cat.mean())
print("Models de Stacking")
print("Precision mitjà (5-fold CV) per stacking_model1:", precision_cv_scores_stk.mean())
print("Precision mitjà (5-fold CV) per stacking_model2:", precision_cv_scores_stk2.mean())
print("Xarxes Neuronals:")
print("Precision mitjà (5-fold CV) per xn1:", precision_cv_scores_xn.mean())
print("Precision mitjà (5-fold CV) per xn2:", precision_cv_scores_xn2.mean())
print("Models de FOP")
print("Accuracy per FOP:", precision_fop)

# Cross Validation amb F1-Score
f1_cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_weighted')
f1_cv_scores_best_rf = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='f1_weighted')
f1_cv_scores_rf_top31 = cross_val_score(rf_top31, X_train_top31, y_train, cv=5, scoring='f1_weighted')
f1_cv_scores_rf_top56 = cross_val_score(rf_top56, X_train_top56, y_train, cv=5, scoring='f1_weighted')

f1_cv_scores_xgb = cross_val_score(xgb, X_train, y_train, cv=5, scoring='f1_weighted')
f1_cv_scores_best_xgb = cross_val_score(best_xgb, X_train, y_train, cv=5, scoring='f1_weighted')
f1_cv_scores_best_xgb2 = cross_val_score(best_xgb2, X_train, y_train, cv=5, scoring='f1_weighted')
f1_cv_scores_best_xgb3 = cross_val_score(best_xgb3, X_train, y_train, cv=5, scoring='f1_weighted')

f1_cv_scores_lgb = cross_val_score(lgb, X_train, y_train, cv=5, scoring='f1_weighted')
f1_cv_scores_best_lgb = cross_val_score(best_lgb, X_train, y_train, cv=5, scoring='f1_weighted')

f1_cv_scores_cat = cross_val_score(cb, X_train, y_train, cv=5, scoring='f1_weighted')
f1_cv_scores_best_cat = cross_val_score(best_cb, X_train, y_train, cv=5, scoring='f1_weighted')

f1_cv_scores_stk = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring='f1_weighted')
f1_cv_scores_stk2 = cross_val_score(stacking_model2, X_train, y_train, cv=5, scoring='f1_weighted')

f1_cv_scores_xn = cross_val_score(xn, X_train, y_train, cv=5, scoring='f1_weighted')
f1_cv_scores_xn2 = cross_val_score(xn2, X_train, y_train, cv=5, scoring='f1_weighted')

f1_fop = f1_score(y_test, y_pred_fop, average='weighted')

print("Models de RandomForest:")
print("F1-Score mitjà (5-fold CV) per rf:", f1_cv_scores_rf.mean())
print("F1-Score mitjà (5-fold CV) per best_rf:", f1_cv_scores_best_rf.mean())
print("F1-Score mitjà (5-fold CV) per rf_top31:", f1_cv_scores_rf_top31.mean())
print("F1-Score mitjà (5-fold CV) per rf_top56:", f1_cv_scores_rf_top56.mean())
print("Models de XGBoost")
print("F1-Score mitjà (5-fold CV) per xgb:", f1_cv_scores_xgb.mean())
print("F1-Score mitjà (5-fold CV) per best_xgb:", f1_cv_scores_best_xgb.mean())
print("F1-Score mitjà (5-fold CV) per best_xgb2:", f1_cv_scores_best_xgb2.mean())
print("F1-Score mitjà (5-fold CV) per best_xgb3:", f1_cv_scores_best_xgb3.mean())
print("Models de LGB")
print("F1-Score mitjà (5-fold CV) per lgb:", f1_cv_scores_lgb.mean())
print("F1-Score mitjà (5-fold CV) per best_lgb:", f1_cv_scores_best_lgb.mean())
print("Models de CatBoost")
print("F1-Score mitjà (5-fold CV) per model_cat:", f1_cv_scores_cat.mean())
print("F1-Score mitjà (5-fold CV) per best_cat:", f1_cv_scores_best_cat.mean())
print("Models de Stacking")
print("F1-Score mitjà (5-fold CV) per stacking_model1:", f1_cv_scores_stk.mean())
print("F1-Score mitjà (5-fold CV) per stacking_model2:", f1_cv_scores_stk2.mean())
print("Xarxes Neuronals:")
print("F1-Score mitjà (5-fold CV) per xn1:", f1_cv_scores_xn.mean())
print("F1-Score mitjà (5-fold CV) per xn2:", f1_cv_scores_xn2.mean())
print("Models de FOP")
print("Accuracy per FOP:", f1_fop)


# Mètrica Personalitzada
se_rf = metric_sobreestimacio(y_test, y_pred_rf)
se_best_rf = metric_sobreestimacio(y_test, y_pred_best_rf)
se_best_rf31 = metric_sobreestimacio(y_test, y_pred_top_rf31)
se_best_rf56 = metric_sobreestimacio(y_test, y_pred_top_rf56)
se_xgb = metric_sobreestimacio(y_test, y_pred_xgb)
se_best_xgb = metric_sobreestimacio(y_test, y_pred_best_xgb)
se_best_xgb2 = metric_sobreestimacio(y_test, y_pred_best_xgb2)
se_best_xgb3 = metric_sobreestimacio(y_test, y_pred_best_xgb3)
se_lgb = metric_sobreestimacio(y_test, y_pred_lgb)
se_best_lgb = metric_sobreestimacio(y_test, y_pred_best_lgb)
se_cb = metric_sobreestimacio(y_test, y_pred_cd)
se_best_cb = metric_sobreestimacio(y_test, y_pred_best_cb)
se_stacking = metric_sobreestimacio(y_test, y_pred_stacking)
se_stacking2 = metric_sobreestimacio(y_test, y_pred_stacking2)
se_xn = metric_sobreestimacio(y_test, y_pred_xn)
se_xn2 = metric_sobreestimacio(y_test, y_pred_xn2)
se_fop = metric_sobreestimacio(y_test, y_pred_fop)

print("Models de RandomForest:")
print("Mètrica personalitzada per rf:", se_rf)
print("Mètrica personalitzada per best_rf:", se_best_rf)
print("Mètrica personalitzada per rf_top31:", se_best_rf31)
print("Mètrica personalitzada per rf_top56:", se_best_rf56)
print("Models de XGBoost")
print("Mètrica personalitzada per xgb:", se_xgb)
print("Mètrica personalitzada per best_xgb:", se_best_xgb)
print("Mètrica personalitzada per best_xgb2:", se_best_xgb2)
print("Mètrica personalitzada per best_xgb3:", se_best_xgb3)
print("Models de LGB")
print("Mètrica personalitzada per lgb:", se_lgb)
print("Mètrica personalitzada per best_lgb:", se_best_lgb)
print("Models de CatBoost")
print("Mètrica personalitzada per model_cat:", se_cb)
print("Mètrica personalitzada per best_cat:", se_best_cb)
print("Models de Stacking")
print("Mètrica personalitzada per stacking_model:", se_stacking)
print("Mètrica personalitzada per stacking_model2:", se_stacking2)
print("Xarxes Neuronals:")
print("FMètrica personalitzada per stacking_model:", se_xn)
print("Mètrica personalitzada per stacking_model2:", se_xn2)
print("Models de FOP")
print("Mètrica personalitzada per stacking_model:", se_fop)