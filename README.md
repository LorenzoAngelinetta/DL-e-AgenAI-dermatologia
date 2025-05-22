
📌 **OBIETTIVO**  
Questo progetto esplora e confronta diversi modelli di Deep Learning e Agent AI multimodali per la diagnosi automatica di immagini dermatologiche. L’obiettivo finale è sviluppare e valutare sistemi capaci di:  
- Classificare correttamente immagini di lesioni cutanee, identificandone la classe di appartenenza.  
- Distinguere tra immagini dermatologiche e immagini generiche.  
- Generare descrizioni testuali coerenti per le immagini classificate come lesioni, includendo la categoria identificata e valutando l’efficacia comunicativa della risposta.  

Questi strumenti mirano a contribuire allo sviluppo di sistemi di supporto clinico basati sull’AI, migliorando l’efficienza diagnostica e rendendo la tecnologia un alleato affidabile nel campo sanitario.

📂 **DATASET**  
- **HAM_TEST2018**: set di valutazione contenente 1511 immagini dermatologiche, suddivise in 7 classi, utilizzato per la valutazione dei modelli classificatori.  
  - Link: https://challenge.isic-archive.com/data/#2018 (include il file.csv con le corrispondenti etichette)  
- **HAM TEST2018 + ImageNetMini1000**: estensione del dataset precedente con 1000 immagini generiche provenienti da ImageNet, arrivando a un totale di 2511 immagini, usate per la valutazione degli Agent AI.  
  - Link: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000  

🛠️ **SETUP**  
Per eseguire il codice, è necessario inserire i percorsi locali a:  
- 📁 Dataset di valutazione  
- 📄 File.csv contenente le etichette delle immagini  
- 📝 File.json in cui memorizzare i risultati delle analisi  

Per utilizzare il modello Gemini 2.0 Flash è necessario utilizzare una API Key personale.  

Librerie generali necessarie per tutti i modelli e componenti del progetto:  
```
pip install torch agno scikit-learn matplotlib seaborn pandas pillow
```  
Le librerie specifiche per ciascun modello sono indicate nelle relative sezioni.

🤖 **MODELLI REALIZZATI**  

**Modelli classificatori:**  
- **ViT-base-HAM10000**: modello Vision Transformer addestrato su 10.000 immagini di lesioni cutanee.  
  - 🔧 Installazione:  
    ```bash
    pip install transformers
    ```  

- **Gemini 2.0 Flash**: modello multimodale sviluppato da Google.  
  - 🔧 Installazione:  
    ```bash
    pip install agno duckduckgo-search
    ```  

- **LLaVA-Med q8**: modello multimodale ottimizzato per il dominio medico.  
  - 🔧 Installazione:  
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    nohup ollama serve &
    ollama pull z-uo/llava-med-v1.5-mistral-7b_q8_0
    pip install agno duckduckgo-search ollama
    ```  

**Agenti AI:**  
- **Agent Gemini con ViT integrato**  
  - 🔧 Installazione:  
    ```bash
    pip install agno transformers
    ```  

- **Agent Gemini con tool ViT esterno**  
  - 🔧 Installazione:  
    ```bash
    pip install agno transformers
    ```  

- **Agent LLaVA-Med con ViT integrato**  
  - 🔧 Installazione:  
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    nohup ollama serve &
    ollama pull z-uo/llava-med-v1.5-mistral-7b_q8_0
    pip install agno ollama transformers
    ```  

📈 **PERFORMACE OTTENUTE**  

| Classificatori       | Accuracy classificazione lesione |
|----------------------|----------------------------------|
| ViT-base-HAM10000    | 74.78%                           |
| Gemini 2.0 Flash     | 32.7%                            |
| LLaVA-Med q8         | 12.5%                            |

I risultati mostrano la superiorità del modello visivo ViT nella classificazione, mentre i modelli linguistici (LLM) sono meno adatti a questo compito.

| Agenti AI                     | Classificazione iniziale | Presenza classificazione in risposta |
|------------------------------|---------------------------|--------------------------------------|
| Gemini con ViT integrato     | 99.56%                    | 99.41%                               |
| Gemini con tool ViT esterno  | 98.56%                    | 98.32%                               |
| LLaVA-Med con ViT integrato  | 81.92%                    | 77.71%                               |

Gli agenti Gemini si sono dimostrati molto efficaci nel distinguere tra immagini dermatologiche e generiche, integrando con coerenza la classificazione nella risposta. L’agente LLaVA-Med, pur con una precisione inferiore, ha generato risposte più ricche di termini medici.
