# Deep Learning e Agent AI in dermatologia

## 📌 OBIETTIVO  
Questo progetto esplora e confronta diversi modelli di Deep Learning e Agent AI multimodali per la diagnosi automatica di immagini dermatologiche. L’obiettivo finale è sviluppare e valutare sistemi capaci di:
- Classificare correttamente immagini di lesioni cutanee, identificandone la classe di appartenenza.
- Distinguere tra immagini dermatologiche e immagini generiche.
- Generare descrizioni testuali coerenti per le immagini classificate come lesioni, includendo la categoria identificata e valutando l’efficacia comunicativa della risposta.
Questi strumenti mirano a contribuire allo sviluppo di sistemi di supporto clinico basati sull’AI, migliorando l’efficienza diagnostica e rendendo la tecnologia un alleato affidabile nel campo sanitario.

## 📂 DATASET 
- **HAM_TEST2018**: set di valutazione contenete 1511 immagini dermatologiche, suddivise in 7 classi, utilizzato per la valutazione dei modelli classificatori  
  - Link: https://challenge.isic-archive.com/data/#2018 (include il file.csv con le corrispondenti etichette)  
- **HAM TEST2018 + ImageNetMini1000**: estensione del dataset precedente con 1000 immagini generiche provenienti da ImageNet, arrivando a un totale di 2511 immagini, usate per la valutazione degli Agent AI.  
  - Link: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000  

## 🛠️ SETUP
Per eseguire il codice, è necessario inserire i percorsi locali a:  
- 📁 Dataset di valutazione  
- 📄 File.csv contenente le etichette delle immagini  
- 📝 File.json in cui memorizzare i risultati delle analisi  

Per utilizzare il modello Gemini 2.0 Flash è necessario utilizzare una API Key personale.  

Le seguenti librerie sono necessarie per tutti i modelli e componenti del progetto. Puoi installarle eseguendo:
```
pip install torch agno scikit-learn matplotlib seaborn pandas pillow
```  
Le librerie specifiche per ciascun modello sono indicate nelle relative sezioni.

## 🤖 MODELLI 
Modelli utilizzati come classificatoridi immagini dermatologiche: 

- **ViT-base-HAM10000**: modello puramente visivo basato su un architettura Vision Transformer e addestrato su un dataset composto da 10.000 immagini di lesioni cutanee.
  - 🔧 Installazione:  
    ```
    pip install transformers
    ```  

- **Gemini 2.0 Flash**: modello multimodale sviluppato da Google capace di svolgere compiti generali.  
  - 🔧 Installazione:  
    ```
    pip install agno duckduckgo-search
    ```  

- **LLaVA-Med q8**: modello multimodale ottimizzato per il dominio medico generale.  
  - 🔧 Installazione:  
    ```
    curl -fsSL https://ollama.com/install.sh | sh sh
    nohup ollama serve &
    ollama pull z-uo/llava-med-v1.5-mistral-7b_q8_0
    pip install agno duckduckgo-search ollama
    ```  

Gli agenti AI sviluppati operano in due fasi: identificazione dell’immagine come lesione cutanea o generica tramite le capacità multimodali dell’LLM e verifica della coerenza tra la diagnosi del classificatore ViT e la risposta generata dall’agente per le immagini classificate come lesioni. 

- **Agent Gemini con ViT integrato**: l’agente segue un percorso fisso valutando se l’immagine è una lesione cutanea e, se confermata, invoca il classificatore (codificato nel flusso) per ottenere una diagnosi utilizzata nella generazione della risposta.  
  - 🔧 Installazione:  
    ```
    pip install agno transformers
    ```  

- **Agent Gemini con tool ViT esterno**: l’agente ha la possibilità di utilizzare il classificatore come tool esterno. Il suo utilizzo avviene in modo autonomo lasciando all’agente la libertà di decidere se ricorrervi o meno in base al contesto, senza seguire un percorso fisso. 
  - 🔧 Installazione:  
    ```
    pip install agno transformers
    ```  

- **Agent LLaVA-Med con ViT integrato**: struttura analoga al primo agente, ma utilizza il modello fine-tunato sul dominio medico per analizzare come cambia la qualità e la precisione delle risposte.
  - 🔧 Installazione:  
    ```
    curl -fsSL https://ollama.com/install.sh | sh
    nohup ollama serve &
    ollama pull z-uo/llava-med-v1.5-mistral-7b_q8_0
    pip install agno ollama transformers
    ```  

## 📈 PERFORMACE OTTENUTE  

| Classificatori       | Accuracy classificazione lesione |
|----------------------|----------------------------------|
| ViT-base-HAM10000    | 74.78%                           |
| Gemini 2.0 Flash     | 32.69%                           |
| LLaVA-Med q8         | 12.51%                           |

I risultati ottenuti dai modelli classificatori evidenziano la buona capacità del modello visivo ViT nell’estrarre e identificare pattern specifici nelle immagini, classificando correttamente circa il 75% delle immagini. Le performance dei modelli Gemini 2.0 Flash e LLaVA-Med q8 sono risultate significativamente inferiori, in quanto si tratta di modelli linguistici progettati principalmente per la comprensione e generazione di linguaggio naturale, non specificamente ottimizzati per la classificazione di immagini.

| Agenti AI                    | Classificazione iniziale (lesione – generica)  | Presenza classificazione in risposta finale |
|------------------------------|------------------------------------------------|---------------------------------------------|
| Gemini con ViT integrato     | 99.56%                                         | 99.41%                                      |
| Gemini con tool ViT esterno  | 98.56%                                         | 98.32%                                      |
| LLaVA-Med con ViT integrato  | 81.92%                                         | 77.71%                                      |

Gli Agenti Gemini, sia con tool ViT esterno che integrato nel flusso operativo, hanno raggiunto performance eccellenti nella distinzione di immagini generiche e dermatologiche dove, per quest’ultime, viene integrata la categorizzazione nella risposta finale in quasi tutte le risposte. L’agente LLaVA-Med si è dimostrato efficace per la classificazione iniziale, seppur con precisione inferiore, generando però risposte molto più precise caratterizzate da termini medici. 
