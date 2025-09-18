# FOO
Food Oracle and Overseer

# Obiettivi
Il progetto FOO – Food Oracle and Overseer nasce dall’esigenza concreta di supportare le persone nelle scelte alimentari quotidiane, offrendo un sistema intelligente e personalizzabile che propone ricette basate su gusti, abitudini e risorse dell’utente.
Gli obiettivi principali sono:
- Supportare l’utente nella scelta del pasto, riducendo il tempo e lo sforzo necessari per prendere decisioni.
- Ottimizzare l’uso degli ingredienti disponibili, riducendo sprechi e migliorando la gestione delle risorse alimentari in casa.
- Offrire raccomandazioni personalizzate, adattandosi progressivamente alle preferenze e ai feedback dell’utente.
- Educare l’utente alla cucina, fornendo ricette dettagliate, chiare e adatte al livello di abilità individuale.
- Promuovere la salute e il benessere, favorendo scelte alimentari più consapevoli e bilanciate.

# Setup
Istruzioni per far partire l'applicativo (Server):
1. clonare il repository e installare le dipendenze (`pip install -r requirements.txt`)
2. creare un file `.env` allo stesso livello di dockerfile. Inserire le informazioni come nel .env-example
3. lanciare il comando `$ docker compose up` (oppure far partire il dockerfile dal pycharm)
4. Al percorso *TagManager/mvp_tagging/full_tagged_dataset_10%.zip* estrarre il file contenuto nel .zip nella stessa cartella.
5. Eseguire i seguenti comandi:
   - `py manage.py loaddata tags.json` - carica le categorie dei tag (TagType) predefiniti.
   - `py manage.py cfr --nrows 100` - carica prime 100 ricette non ancora caricate. Queste sono prese dal file .csv estratto prima. Insieme alle ricette sono popolati anche ingredienti relativi a ogni ricetta.
6. Infine fare `py manage.py runserver`

Istruzioni per far partire corettamente il notebook:
1. Scaricate il [dataset](https://recipenlg.cs.put.poznan.pl/dataset)
2. Inserire il dataset nella cartella *TagManager/mvp_tagging/*
3. Lanciare il `main()` dello script *split_data_set* (si puo cambiare il numero dei chunk in cui il dataset viene sudiviso tramite il parametro nel main)
4. Nella prima cella del notebook `recipes.ipynb` inserire i riferimenti degli chunk con cui effetuare l'addestramento
5. Far partire il notebook
