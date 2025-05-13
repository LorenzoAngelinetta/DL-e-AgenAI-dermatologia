#IMPORTAZIONE LIBRERIE
from agno.tools.duckduckgo import DuckDuckGoTools
import torch
import os
from agno.agent import Agent
from agno.media import Image
from agno.models.ollama import Ollama
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#CARICAMENTO DEL DRIVE
path_valutazione = "Path del dataset di test"
path_csv = "Path del file.csv contenente le etichette per le immagini di test"
json_path = "Path del file.json per il salvataggio delle analisi eseguite"

#ISTRUZIONI E DESCRIZIONE PER L'AGENTE (IN COSA E' SPECIALIZZATO E COSA DEVE FARE)
descrizione = [
    "You are specialized in dermatology. Analyze the image to determine the ONLY ONE the most likely class of skin lesion.",
]

istruzioni = [
    "Look carefully at the image of the skin lesion.",
    "Compare its features (shape, color, texture, borders, size and other features) with the class descriptions.",
    "Choose ONLY one class that seems the most similar.",
    "Respond only with the class name, no extra information.",

    "Class Descriptions:",
    "- Melanocytic nevi (nv): It is the most likely class. are round, smooth, and uniform in color black or brown, typically appearing in shades of brown, black, or pink. They are well-defined, symmetrical, and do not change significantly in size, shape, or color over time. The surface is usually flat or slightly raised, with a consistent and even texture. The borders are sharp and clear, with no irregularities. Nevi are typically small, with a diameter less than 6mm, and remain stable in size throughout life. However, SOME nevi may exhibit mild asymmetry or slight color variation, which could lead to MISCLASSIFICATION as melanoma. Large congenital nevi or those with uneven edges may be particularly challenging for automated models to differentiate. The surrounding skin is typically unaffected, without visible redness or swelling."
    "- Vascular lesions (vasc): appear as red, purple, or blue spots caused by abnormal blood vessels. These lesions can be flat or raised with a smooth texture and are often clustered in small, localized areas. They are common on the face, chest, or limbs. Vascular lesions are typically benign and may be associated with conditions like cherry angiomas or spider veins. In some cases, they may cause discomfort or bleeding if injured."
    "- Actinic keratoses and Bowen’s disease (akiec): present as rough, dry, scaly patches with a red (reddish) hue, commonly found on sun-exposed areas such as the face, ears, hands, or scalp. These lesions can be flat or slightly raised with a crusty surface. Actinic keratosis can feel tender or itchy, and Bowen's disease typically appears as reddish or brownish plaques with well-defined borders. Both conditions are pre-cancerous and can develop into squamous cell carcinoma if left untreated.",
    "- Dermatofibromas (df): are firm, small, brownish nodules that exhibit a central dimpling effect when pressed. These lesions are round and well-defined with smooth or slightly raised surfaces. Dermatofibromas are usually harmless and do not require treatment unless they become painful, irritated, or cosmetically concerning. The dimpling effect upon pressure is a distinguishing feature of this type of lesion.",
    "- Basal cell carcinoma (bcc): lesions appear as shiny, pearly nodules or flat pinkish patches, often surrounded by small visible blood vessels. They have smooth, translucent borders and are most commonly found on sun-exposed areas like the face, neck, or ears. These lesions grow slowly and may have a raised, waxy appearance. While rarely metastatic, untreated BCC can invade surrounding tissues, causing scarring and deformity."
    "- Benign keratosis-like lesions (bkl): are raised growths with a rough, wart-like texture, usually brown or yellowish in color. They are typically round, symmetrical, and commonly found on the chest, back, or limbs. BKLs do not change significantly over time and are benign. However, they can be mistaken for other skin conditions, so periodic monitoring is advised to check for changes in size or appearance.",
    "- Melanoma (mel): lesions are dark, irregularly shaped, and asymmetrical, with jagged or notched borders. They exhibit multiple shades of dark, red, blue, or white and tend to change in size and appearance over time. Melanomas are usually larger than 6mm in diameter, and their size tends to increase over time. These lesions often have a rough or scaly texture, and the surface may be ulcerated or raised unevenly. The borders are blurry or undefined, often showing irregularities. The ABCDE rule (Asymmetry, Border irregularity, Color variation, Diameter larger than 6mm, Evolution over time) is often used to identify melanoma. Additionally, the surrounding skin may show signs of inflammation, redness, or even bleeding, indicating a more aggressive or advanced stage.",
]

#DIZIONARIO DELLE CLASSI PER AVERE IL NOME DELLA CLASSE IDENTIFICATA DA HAM10000
acronimo_classi = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc", 7: "No class"}
acronimo_classi_inverso = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6, "No class": 7}

nomi_estesi = {
    0: ["actinic keratoses", "intraepithelial carcinoma", "Bowen's disease"],
    1: ["basal cell carcinoma"],
    2: ["benign keratosis-like lesions"],
    3: ["dermatofibroma"],
    4: ["melanoma"],
    5: ["melanocytic", "nevi", "mole", "nevus"],
    6: ["vascular lesions"],
    7: ["No class"]
}

#CLASSE AGENTE OLLAMA
class Agent_Ollama:
    def __init__(self): #Costruttore della classe
        self.agent = Agent(
                model=Ollama(id="z-uo/llava-med-v1.5-mistral-7b_q8_0"), #Id LLM utilizzato
                tools = [DuckDuckGoTools()],
                show_tool_calls=True,
                instructions = istruzioni,
                description = descrizione,
                markdown = True,
              )
        self.dati = self.carica_dati() #carica i dati dal file json

    #Metodo per runnare l'agent e ottenere la risposta
    def esegui(self, testo, immagine):
        img = Image(filepath=immagine)  # Carica immagine
        risposta_ollama = self.agent.run(testo, images=[img]).content #risposta ottenuta da LLaVa-med
        return risposta_ollama

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
        df = pd.read_csv(path_csv) #legge csv con etichette corrette
        df = df.set_index("image") #nome dell'immagine sarà l'indice

        img_test = sorted([os.path.join(path_valutazione, img) for img in os.listdir(path_valutazione) if img.lower().endswith(('png', 'jpg', 'jpeg'))]) #lista delle img ordinate
        img_test = [img for img in img_test if img not in [d["img"] for d in self.dati["valutate"]]] #lista togliendo le img già valutate

        count = 0

        for img_path in img_test:
            img_name = os.path.basename(img_path).split('.')[0] #nome immagine senza estensione
            classificazione_llavamed = self.esegui(testo, img_path) #ottengo la risposta dal modello
            classificazione_llavamed_numerica = 7

            for numero in acronimo_classi: #controllo se nella risposta è stata messa la classificazione
                if any(nome in classificazione_llavamed.lower() for nome in nomi_estesi[numero]) or acronimo_classi[numero] in classificazione_llavamed.lower():
                    classificazione_llavamed_numerica = numero
                if all(acronimo_classi[key] in classificazione_llavamed.lower() for key in acronimo_classi if key != max(acronimo_classi)) and acronimo_classi[max(acronimo_classi)] not in classificazione_llavamed.lower():
                    classificazione_llavamed_numerica = 7
                    break

            if img_name in df.index: #verifico classificazione corretta nel csv
                classificazione_reale = df.loc[img_name].idxmax().lower() #Cerca tipo di lesione nel csv per quell'immagine
                classificazione_reale_numerica = acronimo_classi_inverso.get(classificazione_reale) #converte in numero

            etichetta_reale = "lesione cutanea" #sono tutte lesioni cutanee le img di valutazione

            #creazione della struttura per l'analisi dell'immagine da mettere nel file json
            self.dati["valutate"].append({
                "img": img_path,
                "reale": etichetta_reale,
                "classe_giusta": classificazione_reale,
                "classe_giusta_numerica": classificazione_reale_numerica,
                "classificazione_llavamed": classificazione_llavamed,
                "classificazione_llavamed_numerica": classificazione_llavamed_numerica
            })

            self.salva_dati() #salvataggio dell'analisi su quell'img
            count += 1
            print(f"Image: {img_path} - Real: {classificazione_reale_numerica} - LLaVaMed prediction: {classificazione_llavamed_numerica}")

            if count%10 == 0:
              print("Pause fo exit without problem.")
              time.sleep(5)
              print("End pause.")

        print("The list of image is finished. Check the metrics value.")

    #Metodo per calcolare le metriche di valutazione finale: accuracy e matrice di confusione
    def calcola_metriche(self):
        if not self.dati["valutate"]:
            print("No data available for metric calculation.")
            return

        # Estraggo le etichette reali e predette con i nomi delle classi
        reale = [d["classe_giusta_numerica"] for d in self.dati["valutate"]]
        predetto = [d["classificazione_llavamed_numerica"] for d in self.dati["valutate"]]

        # Calcolo le metriche
        accuracy = accuracy_score(reale, predetto)
        conf_matrix = confusion_matrix(reale, predetto)

        # Stampa delle metriche e grafico
        print(f"NUMBER OF IMAGES ANALYZED: {len(reale)}\n")
        print(f"Accuracy classification: {accuracy * 100:.2f}%\n")
        print("Confusion Matrix Classification LLaVaMed:\n")

        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=[acronimo_classi[i] for i in list(acronimo_classi.keys())],
                    yticklabels=[acronimo_classi[i] for i in list(acronimo_classi.keys())])
        plt.xlabel("Predicted")
        plt.ylabel("Real")
        plt.title("Confusion Matrix - Classification Skin Lesion LLaVaMed")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()

#ISTANZIO OGGETTI CLASSE
ollama_agent = Agent_Ollama()

#VALUTAZIONE DEL MODELLO SU UN SET DI IMG DI LESIONI CUTANEE
input_testuale = "What type of skin lesion class is most likely (among akiec, bcc, bkl, df, mel, nv, vasc)? Use the characteristics in the description to identify the best match and choose only one class."
ollama_agent.salvataggio_analisi(path_valutazione, input_testuale)

#STAMPA DELLE METRICHE DI VALUTAZIONE
ollama_agent.calcola_metriche()
