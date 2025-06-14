#IMPORTAZIONE LIBRERIE
from transformers import ViTImageProcessor, ViTForImageClassification 
import torch
import torch.nn.functional as F
import os
from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image
from sklearn.metrics import confusion_matrix 
import time
import json
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

#CARICAMENTO DEL DRIVE
path_valutazione = "Path del dataset di test"
path_csv = "Path del file.csv contenente le etichette per le immagini di test"
json_path = "Path del file.json per il salvataggio delle analisi eseguite"

#CHIAVE API PER UTILIZZARE AGENTE GEMINI
os.environ["GOOGLE_API_KEY"] = "CHIAVE API"

#ISTRUZIONI E DESCRIZIONE PER L'AGENTE (IN COSA E' SPECIALIZZATO E COSA DEVE FARE)
descrizione = [
    "You are an AI agent specialized in dermatology. Your task is to analyze both the image and the description to determine if they refer to a skin lesion .",
]

istruzioni = [
    "If both the image and the description refer to a skin lesion, respond with 'Yes' and provide a detailed explanation of the text and features observed in the image.",
    "If the image refers to a skin lesion but the description does not, suggest the user to ask only a question about the image.",
    "If the description refers to a skin lesion but the image does not, describe the image briefly in few words.",
]

#DIZIONARIO DELLE CLASSI PER AVERE IL NOME DELLA CLASSE IDENTIFICATA DA HAM10000
etichette_classi = {
    0: "Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)",
    1: "Basal cell carcinoma (bcc)",
    2: "Benign keratosis-like lesions (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic nevi (nv)",
    6: "Vascular lesions (vasc)"
}

acronimo_classi = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}

nomi_estesi = {
    0: ["actinic keratoses", "actinic keratosis", "intraepithelial carcinoma", "Bowen's disease"],
    1: ["basal cell carcinoma"],
    2: ["benign keratosis-like lesions"],
    3: ["dermatofibroma", "dermatofibromas"],
    4: ["melanoma"],
    5: ["melanocytic nevi", "mole", "nevus"],
    6: ["vascular lesions"]
}

#CLASSE AGENTE GEMINI
class Agent_Gemini:
    def __init__(self): #Costruttore della classe
        self.agent = Agent(
                model = Gemini(id = "gemini-2.0-flash-exp"), #id AI utilizzata
                tools = [],
                instructions = istruzioni,
                description = descrizione,
                markdown = True,
              )
        self.dati = self.carica_dati() #carica i dati dal file json

    #Controllo con Gemini per vedere se input è coerente per le lesioni cutanee
    def rilevazione_gemini(self, testo, immagine):
        img = Image(filepath = immagine) #apertura img
        risposta = self.agent.run( #risposta del modello ('yes' se input è coerente a lesioni cutanee)
            testo,
            images=[img]
        ).content #estraggo solo risposta da tutti i dettagli
        return risposta

    #Risposta di Gemini utilizzando dati ottenuti da ViT
    def spiegazione_gemini(self, testo, immagine):
        img = Image(filepath = immagine) #apertura img
        risposta = self.agent.run( #risposta del modello utilizzando il testo con la classificazione fornita da ViT
            testo,
            images=[img]
        ).content #estraggo solo risposta da tutti i dettagli
        return risposta

    #Metodo per la classificazione dell'immagine con 'vit-base-HAM-10000-patch-32' e stampa dell'output
    def classificazione_vit(self, immagine):
        nome_classe = etichette_classi[modello_ViT.output_vit(immagine)]  # Conversione della classe
        output_string = f"\nClass predicted by the 'ViT-base-HAM10000' model: {nome_classe}\n"

        return output_string

    #Metodo che controlla se esiste il file di salvataggio valutazione ritornandolo
    def carica_dati(self):
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                return json.load(f)
        return {"valutate": []}

    #Metodo che serve a scrivere la nuova analisi sul file di salvataggio
    def salva_dati(self):
        with open(json_path, "w") as f:
            json.dump(self.dati, f, indent=4)

    #Metodo che salva la risposta ottenuta dal modello su file json
    def salvataggio_analisi(self, path_valutazione, testo):
        df = pd.read_csv(path_csv)
        df = df.set_index("image") #nome dell'immagine sarà l'indice

        img_test = sorted([os.path.join(path_valutazione, img) for img in os.listdir(path_valutazione) if img.lower().endswith(('png', 'jpg', 'jpeg'))]) #lista delle img ordinate
        img_test = [img for img in img_test if img not in [d["img"] for d in self.dati["valutate"]]] #lista togliendo le img già valutate

        count = 0 # serve per fare la pausa di tot secondi (per limite chiamate API a Gemini)
        random.shuffle(img_test) #immagini mischiate

        for img_path in img_test:
            img_name = os.path.basename(img_path).split('.')[0]

            if img_name in df.index: #verifico se è nel csv (quindi se è lesione)
                etichetta_reale = "lesione_cutanea"
            else:
                etichetta_reale = "non_etichettata"

            risposta_gemini = self.rilevazione_gemini(testo, img_path) #se img e testo riguardano lesioni cutanee

            if "Yes" in risposta_gemini:
              pred_gemini = 1
              classe_vit = modello_ViT.output_vit(img_path) #classe ottenuta di ViT
              stringa_ViT = self.classificazione_vit(img_path) #informazioni ottenute da ViT
              input_aggiornato = testo + " The classification model gave me this information. " + stringa_ViT #nuovo input da passare al modello Gemini
              risposta_finale_gemini = self.spiegazione_gemini(input_aggiornato, img_path) #ottenere descrizione finale di Gemini basata sulla predizione ViT
              etichetta_analisi_presente = 1 if acronimo_classi[classe_vit] in risposta_finale_gemini.lower() or any(nome in risposta_finale_gemini.lower() for nome in nomi_estesi[classe_vit]) else 0 #verifica se predizione di ViT nella descrizione di Gemini
              analisi_lesione = acronimo_classi[classe_vit]
            else:
              pred_gemini = 0
              analisi_lesione = ""
              etichetta_analisi_presente = ""
              risposta_finale_gemini = risposta_gemini

              #Si fa questo controllo perchè se viene classificata lesione cutanea da gemini ma non lo è viene elaborata una risposta (vedere matrice confusione)
              for val in acronimo_classi.values():
                  if val in risposta_finale_gemini.lower():
                      analisi_lesione = val
                      etichetta_analisi_presente = 1
                      break

            #creazione della struttura per l'analisi dell'immagine da mettere nel file json
            self.dati["valutate"].append({
                  "img": img_path,
                  "reale": etichetta_reale,
                  "pred_yes_no": pred_gemini,
                  "analisi_lesione": analisi_lesione,
                  "etichetta_analisi_presente": etichetta_analisi_presente,
                  "descrizione_finale": risposta_finale_gemini
              })

            self.salva_dati() #salvataggio dell'analisi su quell'img
            time.sleep(0.6)
            count+=1;

            print(f"Image: {img_path} - Prediction Gemini: {pred_gemini} - Real Label: {etichetta_reale} - Label present in Gemini Answer: {etichetta_analisi_presente}")

            if count % 5 == 0:
                print("\nWaiting for 55 seconds to prevent overload on Gemini API calls......\nIf you want to stop the analysis, do it now to avoid errors.\n")
                time.sleep(55)

        print("The list of image is finished. Check the metrics value.")

    #Metodo per calcolare le metriche di valutazione finale: accuracy e matrice di confusione (auc roc non ha senso su prima chiamata API perchè ho valutazione su un unico set di classi)
    def calcola_metriche(self):
        if not self.dati["valutate"]:
            print("No data available for metric calculation.")
            return

        # Filtra solo le immagini etichettate come "lesione_cutanea"
        dati_lesioni_cutanee = [d for d in self.dati["valutate"] if d["reale"] == "lesione_cutanea"]
        if not dati_lesioni_cutanee:
            print("No skin lesions detected in the data.")
            return

        #Creazione degli array contenti i dati necessari per valutare il modello
        reale = [d["reale"] for d in self.dati["valutate"]]
        pred_yes_no = [d["pred_yes_no"] for d in self.dati["valutate"]]
        etichetta_analisi_presente = [d["etichetta_analisi_presente"] for d in dati_lesioni_cutanee]

        #Mappa per le etichette di classificazione reale
        mappa_etichetta = {"lesione_cutanea": 1, "non_etichettata": 0}
        reale_numerico = [mappa_etichetta.get(etichetta, -1) for etichetta in reale]

        #Somma le previsioni giuste che ha fatto
        corretto_yes_no = sum(
            (etichetta_reale == "lesione_cutanea" and pred_gemini == 1)
            or
            (etichetta_reale == "non_etichettata" and pred_gemini == 0)
            for etichetta_reale, pred_gemini in zip(reale, pred_yes_no)
        )

        #Metriche di valutazione sulla risposta yes/no del modello gemini
        accuracy_yes_no = corretto_yes_no / len(reale) if len(reale) > 0 else 0
        conf_matrix_yes_no = confusion_matrix(reale_numerico, pred_yes_no)

        #Creazione liste per la presenza lesione cutanea
        dati_lesioni_rilevate = [d for d in self.dati["valutate"] if d["pred_yes_no"] == 1]
        presenza_attesa = [1 if d["reale"] == "lesione_cutanea" else 0 for d in dati_lesioni_rilevate]
        presenza_effettiva = [1 if d["etichetta_analisi_presente"] == 1 else 0 for d in dati_lesioni_rilevate]
        tot_analisi = sum(1 for d in self.dati["valutate"] if d["pred_yes_no"] == 1)

        #Metriche per la presenza della classificazione nella risposta
        accuracy_presenza_analisi = sum(1 for p, r in zip(presenza_effettiva, presenza_attesa) if p == r) / len(presenza_attesa)
        conf_matrix_presenza_analisi = confusion_matrix(presenza_attesa, presenza_effettiva)

        #Stampa e graficamento delle metriche
        print(f"NUMBER OF IMAGE ANALIZED: {len(reale)}")
        print("gemini MODEL 'YES' EVALUATION METRICS:")
        print(f"Accuracy: {accuracy_yes_no * 100:.2f} %")
        print("Confusion Matrix:\n", conf_matrix_yes_no)

        plt.figure(figsize=(3, 2))
        sns.heatmap(conf_matrix_yes_no, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Non-Lesion", "Lesion"], yticklabels=["Non-Lesion", "Lesion"])
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix - Yes/No Prediction")
        plt.show()

        print("")
        print(f"NUMBER OF IMAGE ANALIZED (CLASSIFIES AS SKIN LESION): {tot_analisi}")
        print("EVALUATION METRICS ON FINAL RESPONSE OF gemini MODEL IF A SKIN LESION IS BEEN DETECTED (IF CONTAINS VIT CLASSIFICATION):")
        print(f"Accuracy: {accuracy_presenza_analisi * 100:.2f} %")
        print("Confusion Matrix:\n", conf_matrix_presenza_analisi)

        plt.figure(figsize=(3, 2))
        sns.heatmap(conf_matrix_presenza_analisi, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Absent", "Present"], yticklabels=["Absent", "Present"])
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix - Presence prediction in final response")
        plt.show()


#CLASSE MODELLO VIT
class ViT:
    #Costruttore della classe ViT
    def __init__(self):
        self.processore_img = ViTImageProcessor.from_pretrained("ahishamm/vit-base-HAM-10000-patch-32")  # Caricamento processore immagini
        self.model_ham10000 = ViTForImageClassification.from_pretrained("ahishamm/vit-base-HAM-10000-patch-32")  # Caricamento modello

    #Restistuisce output classificazione
    def output_vit(self, immagine):
        from PIL import Image
        img = Image.open(immagine)  # Apertura img
        inputs = self.processore_img(images=img, return_tensors="pt")  # Preprocessa l'immagine prima di passarla al modello restituendo un tensore

        with torch.no_grad():  # Disattivo gradienti per modalità valutazione
            outputs = self.model_ham10000.vit(**inputs)  # Passo img pre-processata (tensore) al modello per ottenere i risultati

        self.logits = self.model_ham10000(**inputs).logits  # Tensore delle probabilità di ciascuna classe
        classe_predizione = torch.argmax(self.logits, dim=-1).item()  # Classe predetta

        return classe_predizione

    #Probabilità di uscita per ogni classe
    def probabilita_vit(self):
        probabilita = F.softmax(self.logits, dim=-1) * 100  # Conversione logits in probabilità con softmax
        return probabilita

#ISTANZIO OGGETTI CLASSE
gemini_agent = Agent_Gemini()
modello_ViT = ViT()

#VALUTAZIONE DEL MODELLO SU UN SET DI IMG DI LESIONI CUTANEE
input_testuale = "What type of skin lesion is it?"
gemini_agent.salvataggio_analisi(path_valutazione, input_testuale)

#STAMPA DELLE METRICHE DI VALUTAZIONE
gemini_agent.calcola_metriche()
