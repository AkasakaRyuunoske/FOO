import ast
import json
import os
import time
import faiss
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


class RecipeRecommender:
    """
    Sistema di raccomandazione per ricette basato su ingredienti.
    Utilizza FAISS (Facebook AI Similarity Search) per la ricerca vettoriale efficiente
    e Word2Vec per la rappresentazione semantica degli ingredienti.

    FAISS: Libreria ottimizzata per la ricerca di similarità in spazi vettoriali ad alta dimensione
    Word2Vec: Modello di machine learning che rappresenta parole come vettori numerici densi
    """

    def __init__(self, model_name='word2vec'):
        """
        Inizializza il sistema di raccomandazione.

        Parametri:
        - model_name: Tipo di modello da utilizzare ('word2vec' di default)

        Attributi della classe:
        - model_type: Specifica il tipo di embedding utilizzato
        - word2vec_model: Istanza del modello Word2Vec addestrato
        - index: Indice FAISS per la ricerca vettoriale veloce
        - df: DataFrame pandas contenente i dati delle ricette
        """
        self.model_type = 'word2vec'
        self.word2vec_model = None  # Modello Word2Vec non ancora inizializzato
        self.index = None  # Indice FAISS non ancora costruito
        self.df = None  # Dataset non ancora caricato

    def load_dataset(self, file_paths, output_path=None):
        """
        Carica e prepara il dataset da uno o più file CSV.

        Parametri:
        - file_paths: Percorso singolo (stringa) o lista di percorsi ai file CSV
        - output_path: Percorso opzionale dove salvare il dataset combinato

        Processo:
        1. Converte percorso singolo in lista se necessario
        2. Carica ogni file CSV mantenendo solo colonne essenziali (title, NER, link)
        3. Rimuove righe con ingredienti mancanti (NER vuoto)
        4. Combina tutti i DataFrame in uno unico
        5. Elimina ricette duplicate basandosi su titolo e ingredienti
        6. Salva il dataset combinato se richiesto

        NER: Named Entity Recognition - contiene gli ingredienti estratti automaticamente
        """
        print("\n=== LOADING DATASET ===")

        # Normalizza input: converte stringa singola in lista
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # Lista per contenere tutti i DataFrame caricati
        dfs = []
        total_recipes = 0

        print(f"Loading {len(file_paths)} dataset file(s)...")

        # Itera attraverso ogni file specificato
        for i, file_path in enumerate(file_paths):
            print(f"[{i + 1}/{len(file_paths)}] Loading {os.path.basename(file_path)}...")
            try:
                start_time = time.time()
                # Carica CSV usando pandas
                df = pd.read_csv(file_path)
                load_time = time.time() - start_time

                # Verifica presenza colonne necessarie e seleziona solo quelle rilevanti
                if all(col in df.columns for col in ["title", "NER", "link"]):
                    df = df[["title", "NER", "link"]]
                    before_count = len(df)

                    # Rimuove righe con campo NER vuoto (senza ingredienti)
                    df.dropna(subset=["NER"], inplace=True)
                    after_count = len(df)

                    dfs.append(df)
                    total_recipes += after_count

                    print(f"  Loaded {after_count} recipes in {load_time:.2f}s")
                    if before_count != after_count:
                        print(f"    (Removed {before_count - after_count} rows with missing ingredients)")
                else:
                    print(f"  Required columns not found in {file_path}")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")

        # Verifica che almeno un dataset sia stato caricato con successo
        if not dfs:
            print("No valid datasets were loaded")
            raise ValueError("No valid datasets were loaded")

        # Combina tutti i DataFrame in uno unico
        print("\nCombining datasets...")
        start_time = time.time()
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(dfs)} files with {len(self.df)} total recipes in {time.time() - start_time:.2f}s")

        # Rimuove ricette duplicate basandosi su titolo e ingredienti
        print("\nRemoving duplicate recipes...")
        start_time = time.time()
        before_count = len(self.df)
        self.df.drop_duplicates(subset=["title", "NER"], inplace=True)
        after_count = len(self.df)
        duplicates_removed = before_count - after_count

        print(f"Removed {duplicates_removed} duplicates in {time.time() - start_time:.2f}s")
        print(f"Final dataset contains {after_count} unique recipes")

        # Salva dataset combinato se richiesto
        if output_path:
            print(f"\nSaving combined dataset to {output_path}...")
            start_time = time.time()
            # Crea directory se non esiste
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            self.df.to_csv(output_path, index=False)
            print(f"Dataset saved in {time.time() - start_time:.2f}s")

        print("=== DATASET LOADING COMPLETE ===\n")
        return self.df

    def _parse_ingredients(self, ner_string):
        """
        Converte la stringa NER in una stringa di ingredienti separati da spazi.

        Parametri:
        - ner_string: Stringa contenente lista Python degli ingredienti

        Processo:
        1. Verifica che l'input sia una stringa
        2. Usa ast.literal_eval per convertire stringa in lista Python sicuramente
        3. Unisce gli ingredienti con spazi per creare testo continuo
        4. Gestisce errori di parsing restituendo stringa vuota

        ast.literal_eval: Funzione sicura per valutare stringhe contenenti letterali Python
        """
        if isinstance(ner_string, str):
            try:
                # Converte stringa rappresentante lista Python in lista reale
                ingredients_list = ast.literal_eval(ner_string)
                # Unisce ingredienti con spazi per creare testo continuo
                return ' '.join(ingredients_list)
            except (ValueError, SyntaxError):
                # Restituisce stringa vuota se parsing fallisce
                return ''
        return ''

    def _create_embeddings(self):
        """
        Crea rappresentazioni vettoriali (embeddings) per tutte le ricette nel dataset.

        Processo principale:
        1. Verifica che il dataset sia caricato
        2. Seleziona metodo di embedding (Word2Vec o SentenceTransformer)
        3. Per Word2Vec: usa metodo specializzato _create_word2vec_embeddings
        4. Per SentenceTransformer: processa ingredienti e genera embeddings

        Embeddings: Rappresentazioni numeriche dense che catturano significato semantico
        """
        print("\n=== CREATING EMBEDDINGS ===")

        if self.df is None:
            print("No dataset loaded")
            raise ValueError("No dataset loaded. Call load_dataset first.")

        # Seleziona metodo di embedding basato su configurazione
        if self.model_type == 'word2vec':
            embeddings = self._create_word2vec_embeddings()
        else:
            # Metodo alternativo con SentenceTransformer (attualmente commentato)
            print("Processing ingredient data...")
            start_time = time.time()
            # Converte liste ingredienti in testo continuo
            self.df['ner_text'] = self.df['NER'].apply(self._parse_ingredients)
            print(f"Processed ingredient data in {time.time() - start_time:.2f}s")

            # Filtra ricette senza ingredienti validi
            print("Filtering out recipes with empty ingredient lists...")
            start_time = time.time()
            before_count = len(self.df)
            self.df = self.df[self.df['ner_text'].str.strip() != '']
            after_count = len(self.df)

            print(
                f"Removed {before_count - after_count} recipes with empty ingredients in {time.time() - start_time:.2f}s")
            print(f"Proceeding with {after_count} valid recipes")

            # Genera embeddings usando SentenceTransformer
            print(f"\nGenerating embeddings for {len(self.df)} recipes...")
            print("(This is typically the most time-consuming step. Please wait...)")
            start_time = time.time()

            # Codifica tutti i testi degli ingredienti in vettori numerici
            embeddings = self.model.encode(
                self.df['ner_text'].tolist(),
                show_progress_bar=True,
                convert_to_numpy=True
            )

            total_time = time.time() - start_time
            recipes_per_second = len(self.df) / total_time if total_time > 0 else 0
            print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
            print(f"Embedding completed in {total_time:.2f}s ({recipes_per_second:.1f} recipes/second)")

        print("=== EMBEDDING CREATION COMPLETE ===\n")
        return embeddings

    def build_index(self, embeddings=None, index_path=None, df_path=None):
        """
        Costruisce indice FAISS per ricerca vettoriale efficiente.

        Parametri:
        - embeddings: Vettori pre-calcolati (opzionale)
        - index_path: Percorso dove salvare l'indice FAISS
        - df_path: Percorso dove salvare il DataFrame

        Processo:
        1. Crea embeddings se non forniti
        2. Inizializza indice FAISS con dimensione appropriata
        3. Aggiunge tutti i vettori all'indice
        4. Salva indice e DataFrame se richiesto

        IndexFlatL2: Tipo di indice FAISS che usa distanza euclidea (L2) per similarità
        """
        print("\n=== BUILDING SEARCH INDEX ===")

        # Crea embeddings se non forniti esternamente
        if embeddings is None:
            print("No pre-computed embeddings provided, creating embeddings...")
            embeddings = self._create_embeddings()
        else:
            print(f"Using provided embeddings with shape {embeddings.shape}")

        # Inizializza indice FAISS
        print("\nInitializing FAISS index...")
        start_time = time.time()
        dimension = embeddings.shape[1]  # Dimensione dei vettori
        # IndexFlatL2: indice che calcola distanza euclidea esatta
        self.index = faiss.IndexFlatL2(dimension)
        print(f"Created IndexFlatL2 with dimension {dimension} in {time.time() - start_time:.2f}s")

        # Aggiunge tutti i vettori embedding all'indice
        print(f"Adding {len(embeddings)} vectors to index...")
        start_time = time.time()
        self.index.add(embeddings)
        add_time = time.time() - start_time
        vectors_per_second = len(embeddings) / add_time if add_time > 0 else 0

        print(f"Added vectors in {add_time:.2f}s ({vectors_per_second:.1f} vectors/second)")
        print(f"Index now contains {self.index.ntotal} vectors")

        # Salva indice FAISS su disco se richiesto
        if index_path:
            print(f"\nSaving FAISS index to {index_path}...")
            start_time = time.time()
            os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else '.', exist_ok=True)
            faiss.write_index(self.index, index_path)
            print(f"Index saved in {time.time() - start_time:.2f}s")

        # Salva DataFrame se richiesto
        if df_path and self.df is not None:
            print(f"Saving DataFrame to {df_path}...")
            start_time = time.time()
            os.makedirs(os.path.dirname(df_path) if os.path.dirname(df_path) else '.', exist_ok=True)
            self.df.to_csv(df_path, index=False)
            print(f"DataFrame saved in {time.time() - start_time:.2f}s")

        print("=== INDEX BUILDING COMPLETE ===\n")
        return self.index

    def load_model(self, index_path, df_path):
        """
        Carica indice FAISS e DataFrame esistenti da disco.

        Parametri:
        - index_path: Percorso dell'indice FAISS salvato
        - df_path: Percorso del DataFrame salvato

        Processo:
        1. Carica indice FAISS binario
        2. Carica DataFrame CSV
        3. Verifica consistenza dei dati caricati
        4. Gestisce errori di caricamento
        """
        print("\n=== LOADING MODEL ===")
        try:
            # Carica indice FAISS da file binario
            print(f"Loading FAISS index from {index_path}...")
            start_time = time.time()
            self.index = faiss.read_index(index_path)
            index_load_time = time.time() - start_time
            print(f"Index loaded in {index_load_time:.2f}s with {self.index.ntotal} vectors")

            # Carica DataFrame da file CSV
            print(f"Loading DataFrame from {df_path}...")
            start_time = time.time()
            self.df = pd.read_csv(df_path)
            df_load_time = time.time() - start_time
            print(f"DataFrame loaded in {df_load_time:.2f}s with {len(self.df)} recipes")

            print("=== MODEL LOADING COMPLETE ===\n")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("=== MODEL LOADING FAILED ===\n")
            return False

    def _get_ingredients_list(self, ner_string):
        """
        Converte stringa NER in lista Python di ingredienti.

        Parametri:
        - ner_string: Stringa contenente rappresentazione lista ingredienti

        Restituisce:
        - Lista di stringhe rappresentanti ingredienti individuali
        - Lista vuota se conversione fallisce
        """
        if isinstance(ner_string, str):
            try:
                # Converte stringa in lista Python usando valutazione sicura
                return ast.literal_eval(ner_string)
            except (ValueError, SyntaxError):
                # Restituisce lista vuota se parsing fallisce
                return []
        return []

    def recommend(self, ingredients, k=10):
        """
        Raccomanda ricette basate su lista di ingredienti forniti.

        Parametri:
        - ingredients: Lista di ingredienti di input
        - k: Numero di raccomandazioni da restituire

        Processo:
        1. Verifica che modello sia inizializzato
        2. Converte ingredienti input in vettore query
        3. Esegue ricerca similarità usando FAISS
        4. Formatta e restituisce risultati ordinati per similarità

        La ricerca trova ricette con ingredienti più simili semanticamente
        """
        print("\n=== SEARCHING FOR RECIPES ===")

        # Verifica che componenti necessari siano inizializzati
        if self.index is None or self.df is None:
            print("Model not built or loaded yet")
            raise ValueError("Model not built or loaded yet")

        print(f"Searching with ingredients: {', '.join(ingredients)}")

        if self.model_type == 'word2vec':
            # Processo specifico per Word2Vec
            if self.word2vec_model is None:
                raise ValueError("Word2Vec model not trained")

            # Normalizza ingredienti input per matching con vocabolario
            clean_ingredients = [ing.strip().lower().replace(' ', '_')
                                 for ing in ingredients]

            vectors = []
            found_ingredients = []

            # Trova vettori per ingredienti presenti nel vocabolario Word2Vec
            for ingredient in clean_ingredients:
                if ingredient in self.word2vec_model.wv:
                    vectors.append(self.word2vec_model.wv[ingredient])
                    found_ingredients.append(ingredient)

            if not vectors:
                print("No matching ingredients found in vocabulary")
                return []

            print(f"Found {len(found_ingredients)} ingredients in vocabulary: {found_ingredients}")

            # Crea vettore query come media dei vettori ingredienti
            query_vector = np.mean(vectors, axis=0).reshape(1, -1).astype('float32')

        else:
            # Processo per SentenceTransformer (metodo alternativo)
            print("Converting ingredients to embedding...")
            start_time = time.time()
            query_text = ' '.join(ingredients)
            query_vector = self.model.encode([query_text], convert_to_numpy=True)
            print(f"Created query embedding in {time.time() - start_time:.4f}s")

        # Esegue ricerca dei k vicini più simili
        k = min(k, len(self.df))  # Limita k al numero massimo di ricette disponibili
        print(f"Searching for top {k} matches among {self.index.ntotal} recipes...")
        start_time = time.time()

        # FAISS restituisce distanze e indici delle ricette più simili
        distances, indices = self.index.search(query_vector, k=k)
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.4f}s")

        # Formatta risultati in struttura leggibile
        print("Formatting results...")
        start_time = time.time()
        results = []

        for i, idx in enumerate(indices[0]):
            if idx < len(self.df):
                row = self.df.iloc[idx]

                # Gestisce ingredienti basandosi su tipo di modello
                if self.model_type == 'word2vec':
                    try:
                        ingredients_list = ast.literal_eval(row['NER'])
                    except (ValueError, SyntaxError):
                        # Fallback: split per virgola se parsing fallisce
                        ingredients_list = row['NER'].split(',') if isinstance(row['NER'], str) else []
                else:
                    ingredients_list = self._get_ingredients_list(row['NER'])

                # Crea dizionario risultato con informazioni complete
                results.append({
                    'score': float(distances[0][i]),  # Punteggio di similarità
                    'title': row['title'],  # Nome ricetta
                    'ingredients': ingredients_list,  # Lista ingredienti
                    'link': row['link'] if 'link' in self.df.columns else 'No link available'  # URL ricetta
                })

        print(f"Formatted {len(results)} results in {time.time() - start_time:.4f}s")
        print("=== SEARCH COMPLETE ===\n")
        return results

    def _create_word2vec_embeddings(self):
        """
        Crea embeddings usando modello Word2Vec addestrato su ingredienti.

        Processo dettagliato:
        1. Estrae e pulisce liste ingredienti da tutte le ricette
        2. Addestra modello Word2Vec su corpus ingredienti
        3. Crea vettori ricetta come media vettori ingredienti componenti
        4. Filtra dataset mantenendo solo ricette con ingredienti validi

        Word2Vec apprende rappresentazioni semantiche dove ingredienti simili
        hanno vettori vicini nello spazio multidimensionale
        """
        print("Creating Word2Vec embeddings...")

        if self.df is None:
            raise ValueError("No dataset loaded. Call load_dataset first.")

        # Prepara corpus per addestramento Word2Vec
        ingredients_lists = []  # Corpus: lista di liste ingredienti
        valid_indices = []  # Indici ricette con ingredienti validi

        # Processa ogni ricetta nel dataset
        for idx, ner_string in enumerate(self.df['NER']):
            try:
                # Converte stringa NER in lista ingredienti
                ingredients = ast.literal_eval(ner_string)
                # Normalizza ingredienti: minuscolo, sostituisce spazi con underscore
                clean_ingredients = [ing.strip().lower().replace(' ', '_')
                                     for ing in ingredients if ing.strip()]

                if clean_ingredients:
                    ingredients_lists.append(clean_ingredients)
                    valid_indices.append(idx)
            except (ValueError, SyntaxError, TypeError):
                # Gestisce casi dove NER non è lista Python valida
                if isinstance(ner_string, str):
                    # Fallback: split per virgola
                    ingredients = [ing.strip().lower().replace(' ', '_')
                                   for ing in ner_string.split(',') if ing.strip()]
                    if ingredients:
                        ingredients_lists.append(ingredients)
                        valid_indices.append(idx)

        print(f"Processed {len(ingredients_lists)} valid recipes for Word2Vec training")

        if not ingredients_lists:
            raise ValueError("No valid ingredients found for Word2Vec training")

        # Addestra modello Word2Vec sul corpus ingredienti
        print("Training Word2Vec model...")
        self.word2vec_model = Word2Vec(
            sentences=ingredients_lists,  # Corpus di addestramento
            vector_size=100,  # Dimensione vettori embedding
            window=5,  # Contesto: parole considerate attorno a target
            min_count=2,  # Frequenza minima per includere ingrediente
            workers=4,  # Thread paralleli per addestramento
            epochs=10  # Numero iterazioni addestramento
        )

        print(f"Word2Vec model trained with vocabulary size: {len(self.word2vec_model.wv)}")

        # Filtra DataFrame mantenendo solo ricette valide
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        # Crea vettori ricetta come media vettori ingredienti
        print("Creating recipe vectors...")
        recipe_vectors = []

        for ingredients_list in ingredients_lists:
            vectors = []

            # Raccoglie vettori per ingredienti presenti in vocabolario
            for ingredient in ingredients_list:
                if ingredient in self.word2vec_model.wv:
                    vectors.append(self.word2vec_model.wv[ingredient])

            if vectors:
                # Media aritmetica dei vettori ingredienti
                avg_vector = np.mean(vectors, axis=0)
                recipe_vectors.append(avg_vector)
            else:
                # Vettore zero se nessun ingrediente trovato
                recipe_vectors.append(np.zeros(100))

        print(f"Created {len(recipe_vectors)} recipe vectors")
        return np.array(recipe_vectors).astype('float32')

    def save_word2vec_model(self, model_path):
        """
        Salva modello Word2Vec addestrato su disco.

        Parametri:
        - model_path: Percorso file dove salvare modello

        Permette riutilizzo modello senza ri-addestramento
        """
        if self.word2vec_model:
            self.word2vec_model.save(model_path)
            print(f"Word2Vec model saved to {model_path}")

    def load_word2vec_model(self, model_path):
        """
        Carica modello Word2Vec precedentemente salvato.

        Parametri:
        - model_path: Percorso file modello salvato

        Restituisce:
        - True se caricamento riuscito, False altrimenti
        """
        try:
            self.word2vec_model = Word2Vec.load(model_path)
            self.model_type = 'word2vec'
            print(f"Word2Vec model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading Word2Vec model: {e}")
            return False


def test_recommender(recommender, test_cases_file=None):
    """
    Testa sistema raccomandazione con combinazioni ingredienti predefinite.

    Parametri:
    - recommender: Istanza RecipeRecommender da testare
    - test_cases_file: File JSON contenente casi di test

    Processo:
    1. Carica casi di test da file JSON o usa predefiniti
    2. Per ogni combinazione ingredienti:
       - Esegue ricerca raccomandazioni
       - Misura tempo esecuzione
       - Mostra risultati formattati
    3. Fornisce statistiche performance

    Utile per validare qualità raccomandazioni e performance sistema
    """
    print("\n=== TESTING RECOMMENDER ===")

    # Carica casi di test da file se disponibile
    if test_cases_file and os.path.exists(test_cases_file):
        with open(test_cases_file, 'r') as f:
            test_cases_dict = json.load(f)
        test_cases = list(test_cases_dict.values())
        print(f"Loaded {len(test_cases)} test ingredient sets from {test_cases_file}")
    else:
        print(f"Test Ingredients file not found: {test_cases_file}")

    # Esegue test per ogni combinazione ingredienti
    for i, ingredients in enumerate(test_cases):
        print(f"\nTest {i + 1}/{len(test_cases)}: {', '.join(ingredients)}")

        # Misura tempo esecuzione raccomandazione
        start_time = time.time()
        results = recommender.recommend(ingredients, k=5)
        elapsed = time.time() - start_time

        print(f"Found {len(results)} matches in {elapsed:.4f} seconds")

        # Mostra risultati formattati
        for j, recipe in enumerate(results):
            print(f"{j + 1}. {recipe['title']}")
            print(f"   Similarity score: {recipe['score']:.4f}")
            print(f"   Ingredients: {', '.join(recipe['ingredients'])}")
            print(f"   Link: {recipe['link']}")
        print("-" * 50)

    print("=== TESTING COMPLETE ===")


def main():
    """
    Funzione principale che orchestraa intero workflow del sistema.

    Workflow completo:
    1. Configura directory e percorsi file
    2. Verifica esistenza modelli salvati
    3. Se modelli esistono: carica da disco
    4. Se modelli non esistono: addestra da zero
    5. Esegue test raccomandazioni

    Gestisce automaticamente persistenza modelli per efficienza
    """
    print("\n====== RECIPE RECOMMENDER SYSTEM ======\n")

    # Crea directory necessarie se non esistenti
    os.makedirs("models", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)

    # Definisce percorsi file e configurazioni
    dataset_path = "dataset"
    index_path = "models/recipes_faiss.index"  # Indice FAISS
    df_path = "models/recipes_dataframe.csv"  # DataFrame ricette
    word2vec_model_path = "models/word2vec_model.bin"  # Modello Word2Vec
    test_cases_file = "validation/test_ingredients.json"  # Casi di test

    # Specifica chunk dataset da utilizzare (2% in questo caso)
    chunks_to_use = [2]
    file_paths = [f"{dataset_path}/recipies_dataset_tagged_chunk_{size}%" for size in chunks_to_use]
    # Filtra solo file esistenti
    file_paths = [f for f in file_paths if os.path.exists(f)]

    # Inizializza sistema raccomandazione
    recommender = RecipeRecommender()

    # Verifica esistenza di tutti i componenti modello salvati
    if (os.path.exists(index_path) and
            os.path.exists(df_path) and
            os.path.exists(word2vec_model_path)):

        print("Loading existing Word2Vec model...")
        # Tenta caricamento modello completo
        if (recommender.load_model(index_path, df_path) and
                recommender.load_word2vec_model(word2vec_model_path)):
            print("Complete model loaded.")
            test_recommender(recommender, test_cases_file=test_cases_file)
        else:
            print("Failed to load complete model. Rebuilding...")
            # Ricostruisce modello se caricamento fallisce
            df = recommender.load_dataset(file_paths)
            recommender.build_index(index_path=index_path, df_path=df_path)
            recommender.save_word2vec_model(word2vec_model_path)
            test_recommender(recommender, test_cases_file=test_cases_file)
    else:
        print("No existing complete model. Building from scratch...")
        # Costruisce modello da zero se non esistono file salvati
        df = recommender.load_dataset(file_paths)
        recommender.build_index(index_path=index_path, df_path=df_path)
        recommender.save_word2vec_model(word2vec_model_path)
        test_recommender(recommender, test_cases_file=test_cases_file)

    print("\n====== RECIPE RECOMMENDER COMPLETE ======")


# Punto di ingresso programma
if __name__ == "__main__":
    main()
