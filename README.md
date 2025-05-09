# Tesi
Tesi di laurea


Per i classificatori installare !pip install transformers



I dataset da dove sono presi


IL DATASET ISIC HA QUESTA 
import pandas as pd
df = pd.read_csv(path_csv)
conteggio_classi = df.iloc[:, 1:].sum()
print(conteggio_classi)

MEL      171.0
NV       908.0
BCC       93.0
AKIEC     43.0
BKL      217.0
DF        44.0
VASC      35.0
