# Anàlisi de la probabilitat d’incendi en habitatges
L'objectiu d'aquest projecte és fer una predicció del grau d'expansió d'un incendi en habitatges. La base de dades amb la que treballem han estat guardades per la _NFIRS_ i conté tots els incidents del 2023 gestionats pels Bombers, la EMS i l'Arson.

## Estructura del repositori
El repositori en el que ens trobem conté totes les eines necessàries per fer l'execució completa del projecte. A continuació es mostra tots els fitxers del repositori. La Base de Dades, com els fitxers de ```python``` per fer el pre-processament de les dades i  l'entrenament del model, com la memòria final del treball:
```
|-- BaseDeDadesNFIRS.txt
|-- ACSST5Y2023.S1901-Data_Cleaned.csv
|-- NetejaBasica.py
|-- One-Hot.py
|-- Models.py
|-- README.md
|-- Memoria.pdf
```

## Execució del Codi
Per executar el codi, el primer que cal fer és descarregar el repositori o clonar-lo i també la Base de Dades que trobareu seguint l'enllaç de ```BaseDeDadesNFIRS.txt```.  Tot seguit ja es podrà executar el fitxer ```NetejaBasica.py``` que ens farà la importació dels fitxers ```*.txt``` i un pre-processament. A continuació executem el fitxer ```One-Hot.py``` que ens acabarà de convertir el nostre ```DataFrame``` en la manera més adient per tal de poder entrenar els diferents models. Per últim passem al fitxer ```Models.py``` entrena els diferents models i fa la comparació entre ells per decidir amb quin ens quedem.

Recordari: assegureu-vos de fer els canvis pertinents en els path dels arxius.
