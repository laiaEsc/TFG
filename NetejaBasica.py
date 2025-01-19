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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

## configuració consola
pd.set_option('display.max_columns', None)  # Mostra totes les columnes
pd.set_option('display.max_rows', None)     # Mostra totes les files
pd.set_option('display.width', 1000)        # Amplia l'amplada per evitar truncaments
sns.set(style="whitegrid")


#################################################################################
###     CONFIGURACIÓ PER LA IMPORTACIÓ DE FITXERS
#################################################################################
# Ruta fitxer
basicincident_path = 'C:\\Users\\laia\\OneDrive\\MatCAD\\4t.2\\TFG\\nfirs_all_incident_pdr_2023\\nfirs_all_incident_pdr_2023\\NFIRS_2023_100124\\basicincident.txt'
incidentaddress_path = 'C:\\Users\\laia\\OneDrive\\MatCAD\\4t.2\\TFG\\nfirs_all_incident_pdr_2023\\nfirs_all_incident_pdr_2023\\NFIRS_2023_100124\\incidentaddress.txt'
fireincident_path = 'C:\\Users\\laia\\OneDrive\\MatCAD\\4t.2\\TFG\\nfirs_all_incident_pdr_2023\\nfirs_all_incident_pdr_2023\\NFIRS_2023_100124\\fireincident.txt'
civiliancasualty_path = 'C:\\Users\\laia\\OneDrive\\MatCAD\\4t.2\\TFG\\nfirs_all_incident_pdr_2023\\nfirs_all_incident_pdr_2023\\NFIRS_2023_100124\\civiliancasualty.txt'

# Especificar el tipus de dades
basicincident_dict = {
    "INCIDENT_KEY": str,      # Identificador únic de l'incident
    "STATE": str,             # Abreviatura de l'estat (ex: "AK")
    "FDID": str,              # Identificador del departament de bombers
    "INC_DATE": int,          # Data de l'incident en format numèric (MMDDYYYY)
    "INC_NO": str,            # Número de l'incident 
    "EXP_NO": int,            # Número d'exposició (pot ser 0)
    "INC_TYPE": str,          # Tipus d'incident
    "PROP_LOSS": float,       # Pèrdues de continent
    "CONT_LOSS": float,       # Pèrdues de contingut
    "PROP_USE": str,          # Us de la propietat
    "DET_ALERT": str,         # Presència de detector de fum 
    "HAZ_REL": str,           # Presència de materials perillosos
    "MIXED_USE": str          # Propietat d'ús mixt
}

incidentaddress_dict = {
    "INCIDENT_KEY": str,      # Identificador únic de l'incident
    "STATE": str,             # Abreviatura de l'estat (ex: "AK")
    "FDID": str,              # Identificador del departament de bombers
    "INC_DATE": int,          # Data de l'incident en format numèric (MMDDYYYY)
    "INC_NO": str,            # Número de l'incident 
    "EXP_NO": int,            # Número d'exposició (pot ser 0)
    "CITY": str,              # Ciutat
    "ZIP5": str               # Codi
}

fireincident_dict = {
    "INCIDENT_KEY": str,      # Identificador únic de l'incident
    "STATE": str,             # Abreviatura de l'estat (ex: "AK")
    "FDID": str,              # Identificador del departament de bombers
    "INC_DATE": int,          # Data de l'incident en format numèric (MMDDYYYY)
    "INC_NO": str,            # Número de l'incident 
    "EXP_NO": int,            # Número d'exposició (pot ser 0)
    "FACT_IGN_1": str,        # Primer factor contribuent a la ignició
    "HEAT_SOURC": str,        # Font de calor que va iniciar l'incendi
    "FIRST_IGN": str,         # Primer material encès
    "TYPE_MAT": str,          # Tipus de material encès
    "FIRE_SPRD": str,         # Extensió de la propagació del foc
    "STRUC_TYPE": str,        # Tipus d'estructura (ex: residencial, comercial)
    "DETECTOR": str,          # Presència de detector de fum 
    "AES_PRES": str,          # Presència de sistema d'extinció automàtica
    "AREA_ORIG": str          # Àrea d'origen del foc (ex: cuina, dormitori)
}

civiliancasualty_dict = {
    "INCIDENT_KEY": str,      # Identificador únic de l'incident
    "STATE": str,             # Abreviatura de l'estat (ex: "AK")
    "FDID": str,              # Identificador del departament de bombers
    "INC_DATE": int,          # Data de l'incident en format numèric (MMDDYYYY)
    "INC_NO": str,            # Número de l'incident
    "EXP_NO": int,            # Número d'exposició (pot ser 0)
    "SEV": str                # Gravetat de la lesió (ex: "L", "M", "H", "F")
}


#################################################################################
###     NETEJA FITXRE BASICINCIDENT
#################################################################################
# Importar fitxer
basicincident_df = pd.read_csv(basicincident_path, delimiter='^', encoding='unicode_escape', dtype=basicincident_dict, usecols=list(basicincident_dict.keys()))

# Primera visualització de les dades
print(basicincident_df.head())
print(basicincident_df.info())
print(basicincident_df.describe())
print(basicincident_df.isnull().sum())

# Ens quedem només amb els incidents que siguin de tipus incendi
basicincident_df = basicincident_df[basicincident_df['INC_TYPE'].astype(int).between(100, 199)]

# Ens quedem només amb els incendis que hagin passat en residencia
basicincident_df = basicincident_df[basicincident_df['PROP_USE'].isin(['419', '429','439', '449', '459', '460', '462', '464', '400'])] # propietats residencials


# Mirem el percentatge de valors NULLs de cada paràmetre
null_percentage = (basicincident_df.isnull().sum() / len(basicincident_df)) * 100
print(null_percentage)

# Eliminem els registre on STATE és NULL perquè en ser clau primària no pot ser NULL
basicincident_df.dropna(subset=['STATE'], inplace=True)


#################################################################################
###     NETEJA FITXRE INCIDENTADRESS
#################################################################################
# Importar fitxer
incidentaddress_df = pd.read_csv(incidentaddress_path, delimiter='^', encoding='unicode_escape', dtype=incidentaddress_dict, usecols=list(incidentaddress_dict.keys()))

# Primera visualització de les dades
print(incidentaddress_df.head())
print(incidentaddress_df.info())
print(incidentaddress_df.describe())

# Mirem el percentatge de valors NULLs de cada paràmetre
print(incidentaddress_df.isnull().sum())
null_percentage = (incidentaddress_df.isnull().sum() / len(incidentaddress_df)) * 100
print(null_percentage)

# Eliminem els registre on STATE és NULL perquè en ser clau primària no pot ser NULL
incidentaddress_df.dropna(subset=['STATE'], inplace=True)

# Eliminem els registre on CITY és NULL
incidentaddress_df.dropna(subset=['CITY'], inplace=True)

# Eliminem els registre on ZIP5 és NULL
incidentaddress_df.dropna(subset=['ZIP5'], inplace=True)


#################################################################################
###     NETEJA FITXRE FIREINCIDENET
#################################################################################
# Importar fitxer
fireincident_df = pd.read_csv(fireincident_path, delimiter='^', encoding='unicode_escape', dtype=fireincident_dict, usecols=list(fireincident_dict.keys()))

# Primera visualització de les dades
print(fireincident_df.head())
print(fireincident_df.info())
print(fireincident_df.describe())

# Mirem el percentatge de valors NULLs de cada paràmetre
print(fireincident_df.isnull().sum())
null_percentage = (fireincident_df.isnull().sum() / len(fireincident_df)) * 100
print(null_percentage)

# Eliminem els registre on STATE és NULL perquè en ser clau primària no pot ser NULL
fireincident_df.dropna(subset=['STATE'], inplace=True)


#################################################################################
###     NETEJA FITXER CIVILIANCASUALITY
#################################################################################
# Importar fitxer
civiliancasualty_df = pd.read_csv(civiliancasualty_path, delimiter='^', encoding='unicode_escape', dtype=civiliancasualty_dict, usecols=list(civiliancasualty_dict.keys()))

# Primera visualització de les dades
print(civiliancasualty_df.head())
print(civiliancasualty_df.info())
print(civiliancasualty_df.describe())

# Recompte de les víctimes per cada incedi
victimsincident_df = (
    civiliancasualty_df.groupby('INCIDENT_KEY')
    .agg(
        TOTAL_VICTIMS=('SEV', 'count'),
        SEV_MINOR=('SEV', lambda x: (x == '1').sum()),
        SEV_MODERATE=('SEV', lambda x: (x == '2').sum()),
        SEV_SEVERE=('SEV', lambda x: (x == '3').sum()),
        SEV_LIFE_THREATENING=('SEV', lambda x: (x == '4').sum()),
        SEV_DEATH=('SEV', lambda x: (x == '5').sum())
    )
    .reset_index()
)

# Comprovació
print(victimsincident_df.head())
print(victimsincident_df.info())
print(victimsincident_df.describe())


#################################################################################
###     FER MERGE DELS FITXERS
#################################################################################
columnes_clau = ['INCIDENT_KEY', 'STATE', 'FDID', 'INC_DATE', 'INC_NO', 'EXP_NO']

df = basicincident_df.merge(incidentaddress_df, on=columnes_clau, how='left')
df = df.merge(fireincident_df, on=columnes_clau, how='left')
df = df.merge(victimsincident_df, on='INCIDENT_KEY', how='left')

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())


#################################################################################
###     GESTIÓ VALORS NULLS DF
#################################################################################
# Mirem el percentatge de valors NULLs de cada paràmetre
null_percentage = (df.isnull().sum() / len(df)) * 100
print(null_percentage)

# Omplir valor NULLs amb 0 per les columnes relacionades amb les víctimes
victim_columns = ['TOTAL_VICTIMS', 'SEV_MINOR', 'SEV_MODERATE', 'SEV_SEVERE',
                  'SEV_LIFE_THREATENING', 'SEV_DEATH']
df[victim_columns] = df[victim_columns].fillna(0)

# Eliminem els registre on CITY és NULL
df.dropna(subset=['CITY'], inplace=True)

# Eliminem els registre on ZIP5 és NULL
df.dropna(subset=['ZIP5'], inplace=True)

# Mirem si els valors NULLs coparteixen files
plt.imshow(df.isna(), aspect='auto')
plt.show()

# Mirem si hi registres duplicats i els eliminem
duplicates = df.duplicated()
print(f"Nombre de registres duplicats: {duplicates.sum()}")
df = df[~duplicates]

# Eliminem els registre on FIRE_SPRD és NULL
df.dropna(subset=['FIRE_SPRD'], inplace=True)

# DETECTOR i DET_ALERT tenen informació molt similar, ens quedem aquell que ens aporta més informació
print((df['DETECTOR'].value_counts()).head(15)) # 94.264 - ens el quedem
print((df['DET_ALERT'].value_counts()).head(15)) # 32.530 - eliminem
df = df.drop(['DET_ALERT'], axis=1)
df.dropna(subset=['DETECTOR'], inplace=True)
df = df[df['DETECTOR'] != 'U']

# Eliminem els registre on FACT_IGN_1 és NULL i és Unknow
print((df['FACT_IGN_1'].value_counts()).head(15))
df.dropna(subset=['FACT_IGN_1'], inplace=True)
df = df[df['FACT_IGN_1'] != 'UU']

# Eliminem els registre on AES_PRES és NULL  i és Unknow
print((df['AES_PRES'].value_counts()).head(15))
df.dropna(subset=['AES_PRES'], inplace=True)
df = df[df['FACT_IGN_1'] != 'UU']

# Eliminem els registre on ZIP5 és '00000' i '0' perquè són valors no vàlids
print((df['ZIP5'].value_counts()).head(15))
df = df[df['ZIP5'] != '00000']
df = df[df['ZIP5'] != '0']

# Eliminem HAZ_REL i MIXED_USE pel seu gran percentatge de valors NULLs
df = df.drop(['HAZ_REL', 'MIXED_USE'], axis=1)

# Substituim els valros NULLs per la categoria 'UU'
print((df['TYPE_MAT'].value_counts()).head(15))
df['TYPE_MAT'] = df['TYPE_MAT'].fillna('UU')

# Eliminem valors NULLs i capem els valors atípics de PROP_LOSS
print(df['PROP_LOSS'].describe())
df.dropna(subset=['PROP_LOSS'], inplace=True)
mean = df['PROP_LOSS'].mean()
std = df['PROP_LOSS'].std()
threshold_upper = mean + 3 * std
df['PROP_LOSS'] = df['PROP_LOSS'].clip(lower=0, upper=threshold_upper)

# Eliminem valors NULLs i capem els valors atípics de CONT_LOSS
print(df['CONT_LOSS'].describe())
df.dropna(subset=['CONT_LOSS'], inplace=True)
mean = df['CONT_LOSS'].mean()
std = df['CONT_LOSS'].std()
threshold_upper = mean + 3 * std
df['CONT_LOSS'] = df['CONT_LOSS'].clip(lower=0, upper=threshold_upper)


'''
plt.imshow(df.isna(), aspect='auto')
plt.show()

sns.histplot(df['TOT_SQ_FT'], bins=50)
plt.show()
'''


#################################################################################
###     CONVERSIÓ DE DADES
#################################################################################
print(df.info())

# Convertir INC_DATE a datetime
df['INC_DATE'] = pd.to_datetime(df['INC_DATE'], format='%m%d%Y')

# Convertir DETECTOR a bool -  N  com a NO, i 1 com a SI
print((df['DETECTOR'].value_counts()).head(10))
df['DETECTOR'] = df['DETECTOR'].apply(lambda x: 1 if x in ['1'] else 0)
df['DETECTOR'] = df['DETECTOR'].astype(bool)

# Convertir AES_PRES a bool -  N i U com a NO, i 1 i 2 com a SI
print((df['AES_PRES'].value_counts()).head(10))
df['AES_PRES'] = df['AES_PRES'].apply(lambda x: 1 if x in ['1', '2'] else 0)
df['AES_PRES'] = df['AES_PRES'].astype(bool)

# Ajuntar per categories AREA_ORIG
print((df['AREA_ORIG'].value_counts()).head(10)) # Area of Fire Origin
def categorize_area_orig(value):
    if value in ['01', '02', '03', '04', '05', '06', '07', '08', '09']:
        return 'Means_Egress'
    elif value in ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19']:
        return 'Assembly'
    elif value in ['20', '21', '22', '23', '24', '25', '26', '27', '28', '29']:
        return 'Function_Areas'
    elif value in ['30', '31', '32', '33', '34', '35', '36', '37', '38', '39']:
        return 'Technical_Processing_Areas'
    elif value in ['40', '41', '42', '43', '44', '45', '46', '47', '48', '49']:
        return 'Storage_Areas'
    elif value in ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59']:
        return 'Service_Areas'
    elif value in ['60', '61', '62', '63', '64', '65', '66', '67', '68', '69']:
        return 'Equipment_Areas'
    elif value in ['70', '71', '72', '73', '74', '75', '76', '77', '78', '79']:
        return 'Structural_Areas'
    elif value in ['80', '81', '82', '83', '84', '85', '86', '87', '88', '89']:
        return 'Transportation_Areas'
    elif value in ['90', '91', '92', '93', '94', '95', '96', '97', '98', '99']:
        return 'Outside_Areas'
    elif value in ['UU']:
        return 'UU'
    else: 
        return 'Other'
df['AREA_ORIG_CATEGORY'] = df['AREA_ORIG'].apply(categorize_area_orig)
print((df['AREA_ORIG_CATEGORY'].value_counts()).head(10))

# Ajuntar per categories HEAT_SOURC
print((df['HEAT_SOURC'].value_counts()).head(10)) # Heat source
def categorize_heat_sourc(value):
    if value in ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19']:
        return 'Operating_Equipment'
    elif value in ['40', '41', '42', '43', '44', '45', '46', '47', '48', '49']:
        return 'Hot_Object'
    elif value in ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59']:
        return 'Explosives'
    elif value in ['60', '61', '62', '63', '64', '65', '66', '67', '68', '69']:
        return 'Open_Flame_or_Smoking'
    elif value in ['70', '71', '72', '73', '74', '75', '76', '77', '78', '79']:
        return 'Chemical'
    elif value in ['80', '81', '82', '83', '84', '85', '86', '87', '88', '89']:
        return 'Another_Fire'
    elif value in ['UU']:
        return 'UU'
    else: 
        return 'Other'
df['HEAT_SOURC_CATEGORY'] = df['HEAT_SOURC'].apply(categorize_heat_sourc)
print((df['HEAT_SOURC_CATEGORY'].value_counts()).head(10))

# Ajuntar per categories FIRST_IGN
print((df['FIRST_IGN'].value_counts()).head(10)) # Item First Ignited
def categorize_first_ign(value):
    if value in ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19']:
        return 'Structural'
    elif value in ['20', '21', '22', '23', '24', '25', '26', '27', '28', '29']:
        return 'Furniture'
    elif value in ['30', '31', '32', '33', '34', '35', '36', '37', '38', '39']:
        return 'Soft_Goods_and_Wearing_Apparel'
    elif value in ['40', '41', '42', '43', '44', '45', '46', '47', '48', '49']:
        return 'Adorment'
    elif value in ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59']:
        return 'Storage_Supplies'
    elif value in ['60', '61', '62', '63', '64', '65', '66', '67', '68', '69']:
        return 'Liquids'
    elif value in ['70', '71', '72', '73', '74', '75', '76', '77', '78', '79']:
        return 'Organic_Materials'
    elif value in ['80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']:
        return 'General_Materials'
    elif value in ['UU']:
        return 'UU'
    else: 
        return 'Other'
df['FIRST_IGN_CATEGORY'] = df['FIRST_IGN'].apply(categorize_first_ign)
print((df['FIRST_IGN_CATEGORY'].value_counts()).head(10))

# Ajuntar per categories TYPE_MAT
print((df['TYPE_MAT'].value_counts()).head(10)) # Type of Material First Ignited
def categorize_type_mat(value):
    if value in ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19']:
        return 'Flammable_Gas'
    elif value in ['20', '21', '22', '23', '24', '25', '26', '27', '28', '29']:
        return 'Combustible_Liquid'
    elif value in ['30', '31', '32', '33', '34', '35', '36', '37', '38', '39']:
        return 'Chemical'
    elif value in ['40', '41', '42', '43', '44', '45', '46', '47', '48', '49']:
        return 'Plastics'
    elif value in ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59']:
        return 'Natural_Product'
    elif value in ['60', '61', '62', '63', '64', '65', '66', '67', '68', '69']:
        return 'Wood'
    elif value in ['70', '71', '72', '73', '74', '75', '76', '77', '78', '79']:
        return 'Textiles'
    elif value in ['80', '81', '82', '83', '84', '85', '86', '87', '88', '89']:
        return 'With Oil'
    elif value in ['UU']:
        return 'UU'
    else: 
        return 'Other'
df['TYPE_MAT_CATEGORY'] = df['TYPE_MAT'].apply(categorize_type_mat)
print((df['TYPE_MAT_CATEGORY'].value_counts()).head(10))

# Ajuntar per categories FACT_IGN_1
print((df['FACT_IGN_1'].value_counts()).head(30)) # Factors Contributing to Ignition
df['FACT_IGN_1'] = df['FACT_IGN_1'].fillna('UU')
def categorize_fact_ign(value):
    if value in ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19']:
        return 'Misuse_Material'
    elif value in ['20', '21', '22', '23', '24', '25', '26', '27', '28', '29']:
        return 'Mechanical_Failure'
    elif value in ['30', '31', '32', '33', '34', '35', '36', '37', '38', '39']:
        return 'Electrical_Failure'
    elif value in ['40', '41', '42', '43', '44', '45', '46', '47', '48', '49']:
        return 'Design_Deficiency'
    elif value in ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59']:
        return 'Operational_Deficiency'
    elif value in ['60', '61', '62', '63', '64', '65', '66', '67', '68', '69']:
        return 'Natural_Condition'
    elif value in ['70', '71', '72', '73', '74', '75', '76', '77', '78', '79']:
        return 'Fire_Spread'
    elif value in ['NN']:
        return 'NN'
    elif value in ['00']:
        return 'Other'
    else: # value = UU
        return 'UU'
df['FACT_IGN_1_CATEGORY'] = df['FACT_IGN_1'].apply(categorize_fact_ign)
print((df['FACT_IGN_1_CATEGORY'].value_counts()).head(30))


#################################################################################
###     CREAR NOVES VARIABLES
#################################################################################
# MONTH
df['MONTH'] = df['INC_DATE'].dt.month
# Agrupar per estacions MONTH
print((df['MONTH'].value_counts()).head(12))
# agrupar per estacions
def categorize_month(value):
    if value in [12, 1, 2]:
        return 'Winter'
    elif value in [3, 4, 5]:
        return 'Spring'
    elif value in [6, 7, 8]:
        return 'Summer'
    else: 
        return 'Autumn'
df['SEASONS'] = df['MONTH'].apply(categorize_month)
print((df['SEASONS'].value_counts()).head(10))

# TOTAL_LOSS
df['TOTAL_LOSS'] = df['PROP_LOSS'] + df['CONT_LOSS']


# MEAN_INCOME depenen del ZIP5
zip_media_income_path = 'C:\\Users\\laia\\OneDrive\\MatCAD\\4t.2\\TFG\\nfirs_all_incident_pdr_2023\\nfirs_all_incident_pdr_2023\\ACSST5Y2023.S1901-Data_Cleaned.csv'
zip_media_income = pd.read_csv(zip_media_income_path, delimiter=',', encoding='unicode_escape', dtype={"GEO_ID": str, "S1901_C01_012M": int})

# Canvi nom columnes
zip_media_income['ZIP5'] = zip_media_income['GEO_ID']
zip_media_income['MEAN_INCOME'] = zip_media_income['S1901_C01_012M']
zip_media_income = zip_media_income.drop(['GEO_ID', 'S1901_C01_012M'], axis=1)

print(zip_media_income.head())

df = df.merge(zip_media_income[['ZIP5', 'MEAN_INCOME']], on='ZIP5', how='left')
df.dropna(subset=['MEAN_INCOME'], inplace=True)

# Comprovació
print(df.info())
print(df['MEAN_INCOME'].describe())


#################################################################################
###     NORMALITZAR DADES
#################################################################################
# Tranformació logarítmica a TOTAL_LOSS
df['LOG_TOTAL_LOSS'] = np.log1p(df['TOTAL_LOSS'])

sns.histplot(df['LOG_TOTAL_LOSS'], bins=50)
plt.show()

# Tranformació logarítmica a MEAN_INCOME
df['LOG_MEAN_INCOME'] = np.log(df['MEAN_INCOME'])

sns.histplot(df['LOG_MEAN_INCOME'], bins=50)
plt.show()


#################################################################################
###     GUARDAR BASE DE DADES
#################################################################################
df.to_csv('C:\\Users\\laia\\OneDrive\\MatCAD\\4t.2\\TFG\\nfirs_all_incident_pdr_2023\\nfirs_all_incident_pdr_2023\\df_cleaned.csv', index=False)
