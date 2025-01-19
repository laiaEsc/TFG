# -*- coding: utf-8 -*-

#################################################################################
###     CONFIGURACIÓ
#################################################################################
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dbfread import DBF
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
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

## configuració consola
pd.set_option('display.max_columns', None)  # Mostra totes les columnes
pd.set_option('display.max_rows', None)     # Mostra totes les files
pd.set_option('display.width', 1000)        # Amplia l'amplada per evitar truncaments
sns.set(style="whitegrid")


#################################################################################
###     CONFIGURACIÓ PER LA IMPORTACIÓ DE FITXERS
#################################################################################
# Ruta fitxer
df_path = 'C:\\Users\\laia\\OneDrive\\MatCAD\\4t.2\\TFG\\nfirs_all_incident_pdr_2023\\nfirs_all_incident_pdr_2023\\df_cleaned.csv'

# Especificar el tipus de dades
df_dict = {
    "INCIDENT_KEY": "object", 
    "STATE": "category", 
    "FDID": "object", 
    "INC_NO": "object", 
    "EXP_NO": "int32", 
    "INC_TYPE": "category", 
    "PROP_LOSS": "float64",  
    "CONT_LOSS": "float64", 
    "PROP_USE": "category", 
    "CITY": "category",
    "ZIP5": "category",
    "AREA_ORIG": "category",
    "HEAT_SOURC": "category",
    "FIRST_IGN": "category",
    "TYPE_MAT": "category",
    "FACT_IGN_1": "category",
    "FIRE_SPRD": "category",
    "STRUC_TYPE": "category",
    "DETECTOR": "bool",
    "AES_PRES": "bool",
    "AREA_ORIG_CATEGORY": "category",
    "HEAT_SOURC_CATEGORY": "category",
    "FIRST_IGN_CATEGORY": "category",
    "TYPE_MAT_CATEGORY": "category",
    "FACT_IGN_1_CATEGORY": "category",
    "MONTH": "int32",
    "SEASONS": "category",
    "TOTAL_LOSS": "float64",
    "MEAN_INCOME": "float64",
    "TOTAL_VICTIMS": "float64",
    "SEV_MINOR": "float64",
    "SEV_MODERATE": "float64",
    "SEV_SEVERE": "float64",
    "SEV_LIFE_THREATENING": "float64",
    "SEV_DEATH": "float64"
}

# Importar fitxer
df = pd.read_csv(df_path, delimiter=',', encoding='unicode_escape', dtype=df_dict, usecols=list(df_dict.keys()))

# Primera visualització de les dades
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())


#################################################################################
###     FER ONE-HOT ENCODING VARIABLES CATEGORIQUES
#################################################################################
# Agrupar per les categories més importats AREA_ORIG_CATEGORY
print((df['AREA_ORIG_CATEGORY'].value_counts()).head(12))
group_area_orig = ['Function_Areas', 'Structural_Areas', 'Storage_Areas', 'Assembly', 'Outside_Areas', 'UU']
df['AREA_ORIG_CATEGORY'] = df['AREA_ORIG_CATEGORY'].apply(
    lambda x: x if x in group_area_orig else 'Other'
)
df['AREA_ORIG'] = df['AREA_ORIG_CATEGORY']

# Agrupar per les categories més importats HEAT_SOURC_CATEGORY
print((df['HEAT_SOURC_CATEGORY'].value_counts()).head(12))
group_heat_sourc = ['Operating_Equipment', 'Open_Flame_or_Smoking', 'UU', 'Hot_Object']
df['HEAT_SOURC_CATEGORY'] = df['HEAT_SOURC_CATEGORY'].apply(
    lambda x: x if x in group_heat_sourc else 'Other'
)
df['HEAT_SOURC'] = df['HEAT_SOURC_CATEGORY']

# Agrupar per les categories més importats FIRST_IGN_CATEGORY
print((df['FIRST_IGN_CATEGORY'].value_counts()).head(12))
group_first_ign = ['Structural', 'UU', 'General_Materials', 'Organic_Materials', 'Furniture', 'Soft_Goods_and_Wearing_Apparel']
df['FIRST_IGN_CATEGORY'] = df['FIRST_IGN_CATEGORY'].apply(
    lambda x: x if x in group_first_ign else 'Other'
)
df['FIRST_IGN'] = df['FIRST_IGN_CATEGORY']

# Agrupar per les categories més importats TYPE_MAT_CATEGORY
print((df['TYPE_MAT_CATEGORY'].value_counts()).head(12))
group_type_mat = ['UU', 'Wood', 'Textiles', 'Plastics', 'Combustible_Liquid']
df['TYPE_MAT_CATEGORY'] = df['TYPE_MAT_CATEGORY'].apply(
    lambda x: x if x in group_type_mat else 'Other'
)
df['TYPE_MAT'] = df['TYPE_MAT_CATEGORY']

# Agrupar per les categories més importats FACT_IGN_1_CATEGORY
print((df['FACT_IGN_1_CATEGORY'].value_counts()).head(12))
group_fact_ing = ['Misuse_Material', 'NN', 'Electrical_Failure', 'Operational_Deficiency']
df['FACT_IGN_1_CATEGORY'] = df['FACT_IGN_1_CATEGORY'].apply(
    lambda x: x if x in group_fact_ing else 'Other'
)
df['FACT_IGN_1'] = df['FACT_IGN_1_CATEGORY']

# Agrupar per les categories més importats STRUC_TYPE
print((df['STRUC_TYPE'].value_counts()).head(12))
group_struct_type = ['1', '2']
df['STRUC_TYPE_CATEGORY'] = df['STRUC_TYPE'].apply(
    lambda x: x if x in group_struct_type else '0'
)
df['STRUC_TYPE'] = df['STRUC_TYPE_CATEGORY']

# Agrupar per les categories més importats PROP_USE
print((df['PROP_USE'].value_counts()).head(12))
group_prop_use = ['419', '429', '449', '439', '459', '460']
df['PROP_USE_CATEGORY'] = df['PROP_USE'].apply(
    lambda x: x if x in group_prop_use else '400'
)
df['PROP_USE'] = df['PROP_USE_CATEGORY']

# Agrupar per les categories més importats INC_TYPE
print((df['INC_TYPE'].value_counts()).head(50))
group_inc_type = ['111', '113', '121', '118', '122', '114', '123', '116', '120']
df['INC_TYPE_CATEGORY'] = df['INC_TYPE'].apply(
    lambda x: x if x in group_inc_type else '100'
)
df['INC_TYPE'] = df['INC_TYPE_CATEGORY']

df['SEASONS_CATEGORY'] = df['SEASONS']

# One-Hot Encoding
df = pd.get_dummies(df, columns=['AREA_ORIG', 'HEAT_SOURC', 'FIRST_IGN', 'TYPE_MAT',
                                 'FACT_IGN_1', 'SEASONS', 'STRUC_TYPE', 'PROP_USE', 'INC_TYPE'])


#################################################################################
###     CHI-SQUARE TEST (per les variables categòriques)
#################################################################################
# quan + gran sigui el valor chi-suare -> variables + fortament relacionades
cols = ['DETECTOR', 'AES_PRES', 'AREA_ORIG_CATEGORY', 'HEAT_SOURC_CATEGORY', 'FIRST_IGN_CATEGORY',
        'TYPE_MAT_CATEGORY', 'FACT_IGN_1_CATEGORY', 'STRUC_TYPE_CATEGORY', 'PROP_USE_CATEGORY',
        'INC_TYPE_CATEGORY', 'SEASONS_CATEGORY']

results = []

for col in cols:
    contingency_table = pd.crosstab(df[col], df['FIRE_SPRD'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    results.append({'Variable': col, 'Chi-square': chi2, 'p-value': p, 'Significant': p < 0.05})

# Convertir els resultats a DataFrame per visualitzar-los
results_df = pd.DataFrame(results)
print(results_df)

# en tenir variables desquilibrades pot ser que el mètode Chi-square hagi exagerat associació amb FIRE_SPRD


#################################################################################
###     CREAR BASE DE DADES
#################################################################################
df_fire_sprd = df.drop(['STATE', 'FDID', 'INC_NO', 'EXP_NO', 'CITY', 'ZIP5',
                        'AREA_ORIG_CATEGORY', 'HEAT_SOURC_CATEGORY', 'FIRST_IGN_CATEGORY',
                        'TYPE_MAT_CATEGORY', 'FACT_IGN_1_CATEGORY', 'MONTH',
                        'STRUC_TYPE_CATEGORY', 'PROP_USE_CATEGORY', 'INC_TYPE_CATEGORY',
                        'SEASONS_CATEGORY', 'PROP_LOSS', 'CONT_LOSS'], axis=1)

# Comprovació base dades
print(df.info())
print(df_fire_sprd.info())

# Analitzem com es distribueix la varialbe objectiu FIRE_SPRD
sns.histplot(df['FIRE_SPRD'], bins=50)
plt.show()
print((df['FIRE_SPRD'].value_counts()).head(12))

# Equilibrem variable objectiu FIRE_SPRD
## convertim en tipus int
df_fire_sprd['FIRE_SPRD'] = df_fire_sprd['FIRE_SPRD'].astype(int)

## eliminem la classe 3 pels pocs valors que tenim en comparació
df_fire_sprd = df_fire_sprd[df_fire_sprd['FIRE_SPRD'] != 3]

## ajuntem les classes de tipus 4 i 5
df_fire_sprd['FIRE_SPRD'] = df_fire_sprd['FIRE_SPRD'].replace({5: 4})

## canviar els nom de les classes perquè ens quedi:
####    - 0: incendi limitat objecte incendiat (valor original 1)
####    - 1: incendi limitat habitació incendiada (valor original 2)
####    - 2: incendi que afecta a tot l'edifici (valor original 4 i 5)
df_fire_sprd['FIRE_SPRD'] = df_fire_sprd['FIRE_SPRD'].replace({1: 0})
df_fire_sprd['FIRE_SPRD'] = df_fire_sprd['FIRE_SPRD'].replace({2: 1})
df_fire_sprd['FIRE_SPRD'] = df_fire_sprd['FIRE_SPRD'].replace({4: 2})

## fem un undersampling
classes = [df_fire_sprd[df_fire_sprd['FIRE_SPRD'] == classe] for classe in df_fire_sprd['FIRE_SPRD'].unique()]
minority_size = min(len(classe) for classe in classes)
resampled_classes = [resample(classe, replace=False, n_samples=minority_size, random_state=42) for classe in classes]
df_fire_sprd = pd.concat(resampled_classes)
df_fire_sprd = df_fire_sprd.sample(frac=1, random_state=42).reset_index(drop=True)

# Comprovació
print((df_fire_sprd['FIRE_SPRD'].value_counts()).head(12))

# Matriu de correlació
data = df_fire_sprd.drop(['INCIDENT_KEY'], axis=1)
corr_matrix = data.corr()
plt.figure(figsize=(60, 45))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Guardar df
df_fire_sprd.to_csv('C:\\Users\\laia\\OneDrive\\MatCAD\\4t.2\\TFG\\nfirs_all_incident_pdr_2023\\nfirs_all_incident_pdr_2023\\df_fire_sprd.csv', index=False)