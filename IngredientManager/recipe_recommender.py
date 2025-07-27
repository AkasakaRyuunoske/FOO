import ast
import json
import os
import time
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union

import faiss
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


# Abstract Interfaces
class DataLoaderInterface(ABC):
    @abstractmethod
    def load_dataset(self, file_paths: Union[str, List[str]], output_path: Optional[str] = None) -> pd.DataFrame:
        pass


class EmbeddingModelInterface(ABC):
    @abstractmethod
    def create_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def encode_query(self, ingredients: List[str]) -> np.ndarray:
        pass


class SearchIndexInterface(ABC):
    @abstractmethod
    def build_index(self, embeddings: np.ndarray) -> Any:
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> tuple:
        pass


class PersistenceInterface(ABC):
    @abstractmethod
    def save_model(self, model: Any, path: str) -> bool:
        pass

    @abstractmethod
    def load_model(self, path: str) -> Any:
        pass


# Implementations
class DataLoader(DataLoaderInterface):
    """Handles all data loading and preprocessing operations."""

    def load_dataset(self, file_paths: Union[str, List[str]], output_path: Optional[str] = None) -> pd.DataFrame:
        print("\nLOADING DATASET")

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
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(dfs)} files with {len(combined_df)} total recipes in {time.time() - start_time:.2f}s")

        # Rimuove ricette duplicate basandosi su titolo e ingredienti
        print("\nRemoving duplicate recipes...")
        start_time = time.time()
        before_count = len(combined_df)
        combined_df.drop_duplicates(subset=["title", "NER"], inplace=True)
        after_count = len(combined_df)
        duplicates_removed = before_count - after_count

        print(f"Removed {duplicates_removed} duplicates in {time.time() - start_time:.2f}s")
        print(f"Final dataset contains {after_count} unique recipes")

        # Salva dataset combinato se richiesto
        if output_path:
            print(f"\nSaving combined dataset to {output_path}...")
            start_time = time.time()
            # Crea directory se non esiste
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            combined_df.to_csv(output_path, index=False)
            print(f"Dataset saved in {time.time() - start_time:.2f}s")

        print("DATASET LOADING COMPLETE\n")
        return combined_df


class Word2VecEmbeddingModel(EmbeddingModelInterface):
    """Handles Word2Vec embeddings creation and query encoding."""

    def __init__(self):
        self.model = None
        self.vector_size = 100

    def create_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Crea embeddings usando modello Word2Vec addestrato su ingredienti.
        """
        print("Creating Word2Vec embeddings...")

        if df is None:
            raise ValueError("No dataset loaded. Call load_dataset first.")

        # Prepara corpus per addestramento Word2Vec
        ingredients_lists = []  # Corpus: lista di liste ingredienti
        valid_indices = []  # Indici ricette con ingredienti validi

        # Processa ogni ricetta nel dataset
        for idx, ner_string in enumerate(df['NER']):
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
        self.model = Word2Vec(
            sentences=ingredients_lists,  # Corpus di addestramento
            vector_size=100,  # Dimensione vettori embedding
            window=5,  # Contesto: parole considerate attorno a target
            min_count=2,  # Frequenza minima per includere ingrediente
            workers=4,  # Thread paralleli per addestramento
            epochs=10  # Numero iterazioni addestramento
        )

        print(f"Word2Vec model trained with vocabulary size: {len(self.model.wv)}")

        # Update df to only include valid recipes (modify in place)
        df.drop(df.index[~df.index.isin(valid_indices)], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Crea vettori ricetta come media vettori ingredienti
        print("Creating recipe vectors...")
        recipe_vectors = []

        for ingredients_list in ingredients_lists:
            vectors = []

            # Raccoglie vettori per ingredienti presenti in vocabolario
            for ingredient in ingredients_list:
                if ingredient in self.model.wv:
                    vectors.append(self.model.wv[ingredient])

            if vectors:
                # Media aritmetica dei vettori ingredienti
                avg_vector = np.mean(vectors, axis=0)
                recipe_vectors.append(avg_vector)
            else:
                # Vettore zero se nessun ingrediente trovato
                recipe_vectors.append(np.zeros(100))

        print(f"Created {len(recipe_vectors)} recipe vectors")
        return np.array(recipe_vectors).astype('float32')

    def encode_query(self, ingredients: List[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Word2Vec model not trained")

        # Normalizza ingredienti input per matching con vocabolario
        clean_ingredients = [ing.strip().lower().replace(' ', '_')
                             for ing in ingredients]

        vectors = []
        found_ingredients = []

        # Trova vettori per ingredienti presenti nel vocabolario Word2Vec
        for ingredient in clean_ingredients:
            if ingredient in self.model.wv:
                vectors.append(self.model.wv[ingredient])
                found_ingredients.append(ingredient)

        if not vectors:
            print("No matching ingredients found in vocabulary")
            raise ValueError("No matching ingredients found in vocabulary")

        print(f"Found {len(found_ingredients)} ingredients in vocabulary: {found_ingredients}")

        # Crea vettore query come media dei vettori ingredienti
        query_vector = np.mean(vectors, axis=0).reshape(1, -1).astype('float32')
        return query_vector

    def save_model(self, model_path: str):
        """Salva modello Word2Vec addestrato su disco."""
        if self.model:
            self.model.save(model_path)
            print(f"Word2Vec model saved to {model_path}")

    def load_model(self, model_path: str) -> bool:
        """Carica modello Word2Vec precedentemente salvato."""
        try:
            self.model = Word2Vec.load(model_path)
            print(f"Word2Vec model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading Word2Vec model: {e}")
            return False


class FAISSSearchIndex(SearchIndexInterface):
    """Handles FAISS index operations."""

    def __init__(self):
        self.index = None

    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        print("\nBUILDING SEARCH INDEX")

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
        print("INDEX BUILDING COMPLETE\n")

        return self.index

    def search(self, query_vector: np.ndarray, k: int) -> tuple:
        if self.index is None:
            raise ValueError("Index not built")

        # Esegue ricerca dei k vicini più simili
        print(f"Searching for top {k} matches among {self.index.ntotal} recipes...")
        start_time = time.time()

        # FAISS restituisce distanze e indici delle ricette più simili
        distances, indices = self.index.search(query_vector, k=k)
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.4f}s")

        return distances, indices


class ModelPersistence(PersistenceInterface):
    """Handles saving and loading of models and data."""

    def save_model(self, model: Any, path: str) -> bool:
        try:
            start_time = time.time()
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

            if isinstance(model, faiss.IndexFlatL2):
                faiss.write_index(model, path)
                print(f"FAISS index saved to {path} in {time.time() - start_time:.2f}s")
            elif isinstance(model, Word2Vec):
                model.save(path)
                print(f"Word2Vec model saved to {path} in {time.time() - start_time:.2f}s")
            elif isinstance(model, pd.DataFrame):
                model.to_csv(path, index=False)
                print(f"DataFrame saved to {path} in {time.time() - start_time:.2f}s")
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")

            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, path: str) -> Any:
        try:
            start_time = time.time()
            if path.endswith('.index'):
                model = faiss.read_index(path)
                print(f"FAISS index loaded from {path} in {time.time() - start_time:.2f}s with {model.ntotal} vectors")
            elif path.endswith('.bin'):
                model = Word2Vec.load(path)
                print(f"Word2Vec model loaded from {path} in {time.time() - start_time:.2f}s")
            elif path.endswith('.csv'):
                model = pd.read_csv(path)
                print(f"DataFrame loaded from {path} in {time.time() - start_time:.2f}s with {len(model)} recipes")
            else:
                raise ValueError(f"Unsupported file type: {path}")
            return model
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return None


# Main Class
class RecipeRecommender:
    """
    Sistema di raccomandazione per ricette basato su ingredienti.
    Utilizza FAISS (Facebook AI Similarity Search) per la ricerca vettoriale efficiente
    e Word2Vec per la rappresentazione semantica degli ingredienti.

    FAISS: Libreria ottimizzata per la ricerca di similarità in spazi vettoriali ad alta dimensione
    Word2Vec: Modello di machine learning che rappresenta parole come vettori numerici densi
    """

    def __init__(self,
                 model_name='word2vec',
                 data_loader: DataLoaderInterface = None,
                 embedding_model: EmbeddingModelInterface = None,
                 search_index: SearchIndexInterface = None,
                 persistence: PersistenceInterface = None):
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

        # Dependency Injection with defaults
        self.data_loader = data_loader or DataLoader()
        self.embedding_model = embedding_model or Word2VecEmbeddingModel()
        self.search_index = search_index or FAISSSearchIndex()
        self.persistence = persistence or ModelPersistence()

        # Backward compatibility attributes
        self.word2vec_model = None  # Will point to embedding_model.model
        self.index = None  # Will point to search_index.index
        self.df = None  # Dataset

    def load_dataset(self, file_paths: Union[str, List[str]], output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Carica e prepara il dataset da uno o più file CSV.
        """
        self.df = self.data_loader.load_dataset(file_paths, output_path)
        return self.df

    def _parse_ingredients(self, ner_string):
        """
        Converte la stringa NER in una stringa di ingredienti separati da spazi.
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
        """
        print("\nCREATING EMBEDDINGS")

        if self.df is None:
            print("No dataset loaded")
            raise ValueError("No dataset loaded. Call load_dataset first.")

        embeddings = self.embedding_model.create_embeddings(self.df)
        self.word2vec_model = self.embedding_model.model  # Backward compatibility

        print("EMBEDDING CREATION COMPLETE\n")
        return embeddings

    def build_index(self, embeddings=None, index_path=None, df_path=None):
        """
        Costruisce indice FAISS per ricerca vettoriale efficiente.
        """
        print("\nBUILDING SEARCH INDEX")

        # Crea embeddings se non forniti esternamente
        if embeddings is None:
            print("No pre-computed embeddings provided, creating embeddings...")
            embeddings = self._create_embeddings()
        else:
            print(f"Using provided embeddings with shape {embeddings.shape}")

        # Build the index
        self.index = self.search_index.build_index(embeddings)

        # Salva indice FAISS su disco se richiesto
        if index_path:
            print(f"\nSaving FAISS index to {index_path}...")
            self.persistence.save_model(self.index, index_path)

        # Salva DataFrame se richiesto
        if df_path and self.df is not None:
            print(f"Saving DataFrame to {df_path}...")
            self.persistence.save_model(self.df, df_path)

        return self.index

    def load_model(self, index_path, df_path):
        """
        Carica indice FAISS e DataFrame esistenti da disco.
        """
        print("\nLOADING MODEL")
        try:
            # Carica indice FAISS da file binario
            print(f"Loading FAISS index from {index_path}...")
            self.index = self.persistence.load_model(index_path)
            if self.index is None:
                print("MODEL LOADING FAILED\n")
                return False
            self.search_index.index = self.index

            # Carica DataFrame da file CSV
            print(f"Loading DataFrame from {df_path}...")
            self.df = self.persistence.load_model(df_path)
            if self.df is None:
                print("MODEL LOADING FAILED\n")
                return False

            print("MODEL LOADING COMPLETE\n")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("MODEL LOADING FAILED\n")
            return False

    def _get_ingredients_list(self, ner_string):
        """
        Converte stringa NER in lista Python di ingredienti.
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
        """
        print("\nSEARCHING FOR RECIPES")

        # Verifica che componenti necessari siano inizializzati
        if self.search_index.index is None or self.df is None:
            print("Model not built or loaded yet")
            raise ValueError("Model not built or loaded yet")

        print(f"Searching with ingredients: {', '.join(ingredients)}")

        try:
            # Converte ingredienti input in vettore query
            query_vector = self.embedding_model.encode_query(ingredients)

            # Esegue ricerca dei k vicini più simili
            k = min(k, len(self.df))  # Limita k al numero massimo di ricette disponibili
            distances, indices = self.search_index.search(query_vector, k=k)

            # Formatta risultati in struttura leggibile
            print("Formatting results...")
            start_time = time.time()
            results = []

            for i, idx in enumerate(indices[0]):
                if idx < len(self.df):
                    row = self.df.iloc[idx]

                    # Gestisce ingredienti
                    try:
                        ingredients_list = ast.literal_eval(row['NER'])
                    except (ValueError, SyntaxError):
                        # Fallback: split per virgola se parsing fallisce
                        ingredients_list = row['NER'].split(',') if isinstance(row['NER'], str) else []

                    # Crea dizionario risultato con informazioni complete
                    results.append({
                        'score': float(distances[0][i]),  # Punteggio di similarità
                        'title': row['title'],  # Nome ricetta
                        'ingredients': ingredients_list,  # Lista ingredienti
                        'link': row['link'] if 'link' in self.df.columns else 'No link available'  # URL ricetta
                    })

            print(f"Formatted {len(results)} results in {time.time() - start_time:.4f}s")
            print("SEARCH COMPLETE\n")
            return results

        except Exception as e:
            print(f"Error during recommendation: {e}")
            return []

    def save_word2vec_model(self, model_path):
        """
        Salva modello Word2Vec addestrato su disco.
        """
        if hasattr(self.embedding_model, 'model') and self.embedding_model.model:
            self.embedding_model.save_model(model_path)

    def load_word2vec_model(self, model_path):
        """
        Carica modello Word2Vec precedentemente salvato.
        """
        result = self.embedding_model.load_model(model_path)
        if result:
            self.word2vec_model = self.embedding_model.model  # Backward compatibility
        return result


class UserQueryHandler:
    """
    Gestore delle query utente per il sistema di raccomandazione ricette.
    Integra perfettamente con il RecipeRecommender esistente.
    """

    def __init__(self, recommender):
        """
        Inizializza il gestore con un'istanza di RecipeRecommender.

        Parametri:
        - recommender: Istanza RecipeRecommender già inizializzata
        """
        self.recommender = recommender

    def process_user_ingredients_strict(self, user_ingredients, n_results=5):
        """
        Versione più rigorosa che privilegia match esatti degli ingredienti.
        """
        # Get more results initially
        results = self.recommender.recommend(user_ingredients, k=n_results * 3)

        # Re-score based on exact ingredient matches
        scored_results = []
        search_ingredients = [ing.lower().strip() for ing in user_ingredients]

        for recipe in results:
            recipe_ingredients = [ing.lower().strip() for ing in recipe['ingredients']]

            # Count exact matches
            exact_matches = 0
            for search_ing in search_ingredients:
                for recipe_ing in recipe_ingredients:
                    if search_ing == recipe_ing or search_ing in recipe_ing:
                        exact_matches += 1
                        break

            # Combine semantic score with exact match bonus
            combined_score = recipe['score'] - (exact_matches * 2.0)  # Lower is better

            scored_results.append({
                **recipe,
                'combined_score': combined_score,
                'exact_matches': exact_matches
            })

        # Sort by combined score and return top N
        scored_results.sort(key=lambda x: x['combined_score'])
        return self._format_results(scored_results[:n_results])

    def process_user_ingredients(self, user_ingredients, n_results=5):
        """
        Processa gli ingredienti inseriti dall'utente e restituisce raccomandazioni.

        Parametri:
        - user_ingredients: Lista o stringa di ingredienti dell'utente
        - n_results: Numero di risultati da restituire (default: 5)

        Processo:
        1. Normalizza input utente
        2. Esegue embedding sull'input
        3. Fa query al FAISS
        4. Restituisce N risultati migliori

        Restituisce:
        - Lista di dizionari con ricette raccomandate
        """
        print("\nPROCESSING USER QUERY")

        # 1. Normalizza input utente
        if isinstance(user_ingredients, str):
            # Se stringa, divide per virgola e pulisce
            ingredients_list = [ing.strip() for ing in user_ingredients.split(',') if ing.strip()]
        else:
            # Se già lista, usa direttamente
            ingredients_list = user_ingredients

        print(f"User ingredients: {', '.join(ingredients_list)}")
        print(f"Requesting {n_results} recommendations")

        try:
            results = self.recommender.recommend(ingredients_list, k=n_results)

            if results:
                print(f"Found {len(results)} matching recipes")
                return self._format_results(results)
            else:
                print("No recipes found matching the ingredients")
                return []

        except Exception as e:
            print(f"Error processing query: {e}")
            return []

    def _format_results(self, results):
        """
        Formatta i risultati per una presentazione user-friendly.

        Parametri:
        - results: Lista risultati dal recommender

        Restituisce:
        - Lista formattata di raccomandazioni
        """
        formatted_results = []

        for i, recipe in enumerate(results, 1):
            formatted_recipe = {
                'rank': i,
                'title': recipe['title'],
                'similarity_score': round(recipe['score'], 4),
                'ingredients': recipe['ingredients'],
                'ingredients_count': len(recipe['ingredients']),
                'link': recipe['link']
            }
            formatted_results.append(formatted_recipe)

        return formatted_results

    def display_recommendations(self, results):
        """
        Mostra le raccomandazioni in formato leggibile.

        Parametri:
        - results: Lista risultati formattati
        """
        if not results:
            print("No recommendations to display.")
            return

        print(f"\n=== TOP {len(results)} RECIPE RECOMMENDATIONS ===")
        print("-" * 60)

        for recipe in results:
            print(f"{recipe['rank']}. {recipe['title']}")
            print(f"   Similarity Score: {recipe['similarity_score']}")
            print(f"   Ingredients ({recipe['ingredients_count']}): {', '.join(recipe['ingredients'][:5])}")
            if len(recipe['ingredients']) > 5:
                print(f"   ... and {len(recipe['ingredients']) - 5} more")
            print(f"   Link: {recipe['link']}")
            print("-" * 60)


def quick_test():
    """
    Test rapido per verificare che il sistema funzioni.
    """
    print("\nQUICK SYSTEM TEST")

    # Inizializza il sistema
    recommender = RecipeRecommender()

    # Percorsi dei file del modello
    index_path = "models/recipes_faiss.index"
    df_path = "models/recipes_dataframe.csv"
    word2vec_model_path = "models/word2vec_model.bin"

    # Verifica se i file del modello esistono
    if not all(os.path.exists(path) for path in [index_path, df_path, word2vec_model_path]):
        print("Model files not found. Building model first...")
        # Usa il codice esistente per costruire il modello
        dataset_path = "dataset"
        chunks_to_use = [2]
        file_paths = [f"{dataset_path}/recipies_dataset_tagged_chunk_{size}%.csv" for size in chunks_to_use]
        file_paths = [f for f in file_paths if os.path.exists(f)]

        if not file_paths:
            print("ERROR: No dataset files found!")
            print("Please ensure you have dataset files in the 'dataset' folder")
            return False

        df = recommender.load_dataset(file_paths)
        recommender.build_index(index_path=index_path, df_path=df_path)
        recommender.save_word2vec_model(word2vec_model_path)

    # Carica il modello
    print("Loading model...")
    if not (recommender.load_model(index_path, df_path) and
            recommender.load_word2vec_model(word2vec_model_path)):
        print("ERROR: Failed to load model!")
        return False

    # Inizializza il gestore delle query utente
    query_handler = UserQueryHandler(recommender)

    # Test con ingredienti semplici
    test_ingredients = ["milk", "flour", "chocolate"]

    print(f"Testing with ingredients: {test_ingredients}")

    try:

        results = query_handler.process_user_ingredients(test_ingredients, n_results=3)

        if results:
            print("SUCCESS! System is working!")
            print(f"Found {len(results)} recipe recommendations:")

            for recipe in results:
                print(f"  - {recipe['title']} (Score: {recipe['similarity_score']})")

            return True
        else:
            print("No results found. This might indicate an issue.")
            return False

    except Exception as e:
        print(f"ERROR during testing: {e}")
        return False


def test_user_input():
    """
    Test interattivo per provare il sistema manualmente.
    """
    print("\nINTERACTIVE TEST")
    print("This will test the exact workflow you requested:")
    print("1. User inputs ingredients")
    print("2. System performs embedding")
    print("3. System queries FAISS")
    print("4. System returns N best results")
    print("-" * 50)

    recommender = RecipeRecommender()
    index_path = "models/recipes_faiss.index"
    df_path = "models/recipes_dataframe.csv"
    word2vec_model_path = "models/word2vec_model.bin"

    if not (recommender.load_model(index_path, df_path) and
            recommender.load_word2vec_model(word2vec_model_path)):
        print("Model not found or failed to load!")
        return

    query_handler = UserQueryHandler(recommender)

    test_cases = [
        ["chicken", "rice", "vegetables"],
        ["pasta", "tomato", "garlic"],
        ["beef", "potato", "onion", "carrot"],
        ["salmon", "lemon", "dill"],  # More specific
        ["tuna", "olive_oil", "garlic"],  # Mediterranean style
        ["cod_fillets", "lemon", "parsley"],  # Classic preparation
        ["fish_fillets", "butter", "herbs"]  # General fish dish
    ]

    for i, ingredients in enumerate(test_cases, 1):
        print(f"\nTEST {i}")
        print(f"Input ingredients: {ingredients}")
        results = query_handler.process_user_ingredients(ingredients, n_results=5)

        if results:
            print(f"SUCCESS! Found {len(results)} recommendations:")
            for j, recipe in enumerate(results[:3], 1):
                print(f"  {j}. {recipe['title']}")
                print(f"     Score: {recipe['similarity_score']}")
                print(f"     Ingredients: {', '.join(recipe['ingredients'][:4])}...")
        else:
            print("No results found")
        print("-" * 30)


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
    print("\nTESTING RECOMMENDER")

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

    print("TESTING COMPLETE")


def main():
    """
    Funzione principale che orchestra l'intero workflow del sistema.

    Workflow completo:
    1. Configura directory e percorsi file
    2. Verifica esistenza modelli salvati
    3. Se modelli esistono: carica da disco
    4. Se modelli non esistono: addestra da zero
    5. Esegue test raccomandazioni

    Gestisce automaticamente persistenza modelli per efficienza
    """
    print("\nRECIPE RECOMMENDER SYSTEM\n")

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
    file_paths = [f"{dataset_path}/recipies_dataset_tagged_chunk_{size}%.csv" for size in chunks_to_use]
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

    print("\nRECIPE RECOMMENDER COMPLETE")


# Punto di ingresso programma
# if __name__ == "__main__":
#     main()

# Uncomment one of these to run different tests:
# if __name__ == "__main__":
#     quick_test()

if __name__ == "__main__":
    test_user_input()
