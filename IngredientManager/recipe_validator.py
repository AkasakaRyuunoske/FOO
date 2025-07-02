import pandas as pd
import numpy as np
import json
import os
import time
import ast
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Importazione dalla stessa directory del modulo recommender
from recipe_recommender import RecipeRecommender


class RecipeValidator:
    """
    Classe per la validazione e il testing di un sistema di raccomandazione ricette.

    Questa classe implementa una suite completa di test per valutare:
    - La rilevanza delle raccomandazioni rispetto agli ingredienti forniti
    - La presenza di contenuti inappropriati nelle ricette
    - La coerenza semantica tra le raccomandazioni multiple
    - Le prestazioni temporali del sistema
    - La copertura degli ingredienti nel dataset
    """

    def __init__(self, ingredients_file=None, inappropriate_terms_file=None):
        """
        Inizializza il validatore con liste di ingredienti e filtri per contenuti inappropriati.

        Args:
            ingredients_file (str, optional): Percorso al file JSON contenente le liste di ingredienti per i test.
                                            Il file deve contenere dizionari con nomi di test come chiavi e
                                            liste di ingredienti come valori.
            inappropriate_terms_file (str, optional): Percorso al file di testo contenente termini inappropriati
                                                     da filtrare nelle raccomandazioni, uno per riga.
        """
        # Istanza del sistema di raccomandazione che verrà caricata successivamente
        self.recommender = None

        # Carica gli ingredienti di test dal file JSON specificato
        self.test_ingredients = self._load_ingredients(ingredients_file)

        # Carica la lista dei termini inappropriati dal file di testo
        self.inappropriate_terms = self._load_inappropriate_terms(inappropriate_terms_file)

        # Lista per memorizzare i risultati delle validazioni eseguite
        self.validation_results = []

    def _load_ingredients(self, file_path):
        """
        Carica i set di ingredienti per i test da un file JSON.

        Il file JSON deve avere la struttura:
        {
            "nome_test_1": ["ingrediente1", "ingrediente2", ...],
            "nome_test_2": ["ingrediente3", "ingrediente4", ...],
            ...
        }

        Args:
            file_path (str): Percorso al file JSON contenente gli ingredienti di test.

        Returns:
            dict: Dizionario contenente i set di ingredienti per i test.
                 Se il file non esiste o non può essere caricato, ritorna un dizionario vuoto.
        """
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    ingredients = json.load(f)
                print(f"Caricati {len(ingredients)} set di ingredienti di test da {file_path}")
                return ingredients
            except Exception as e:
                print(f"Errore nel caricamento del file ingredienti: {e}")
        else:
            print(f"File di ingredienti di test non trovato: {file_path}")
        return {}

    def _load_inappropriate_terms(self, file_path):
        """
        Carica i termini inappropriati da un file di testo.

        Il file deve contenere un termine per riga. I termini vengono convertiti in minuscolo
        per facilitare la ricerca case-insensitive.

        Args:
            file_path (str): Percorso al file di testo contenente i termini inappropriati.

        Returns:
            list: Lista dei termini inappropriati in minuscolo.
                 Se il file non esiste o non può essere caricato, ritorna una lista vuota.
        """
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    # Legge ogni riga, rimuove spazi bianchi e converte in minuscolo
                    terms = [line.strip().lower() for line in f if line.strip()]
                print(f"Caricati {len(terms)} termini inappropriati da {file_path}")
                return terms
            except Exception as e:
                print(f"Errore nel caricamento del file termini inappropriati: {e}")
        else:
            print(f"File termini inappropriati non trovato: {file_path}")
        return []

    def load_recommender(self, index_path, df_path):
        """
        Carica un indice FAISS esistente e il DataFrame associato per il sistema di raccomandazione.

        Questo metodo inizializza il sistema di raccomandazione caricando:
        1. L'indice FAISS precomputato per la ricerca vettoriale
        2. Il DataFrame contenente i dati delle ricette
        3. Il modello Word2Vec per l'elaborazione del linguaggio naturale

        Args:
            index_path (str): Percorso al file dell'indice FAISS (.index).
            df_path (str): Percorso al file CSV contenente il DataFrame delle ricette.

        Returns:
            bool: True se il caricamento è andato a buon fine, False altrimenti.
        """
        # Crea una nuova istanza del sistema di raccomandazione
        self.recommender = RecipeRecommender()

        # Carica il modello base (indice FAISS e DataFrame)
        success = self.recommender.load_model(index_path, df_path)

        if success:
            # Verifica se il modello Word2Vec è già caricato nell'istanza
            if not hasattr(self.recommender, 'word2vec_model') or self.recommender.word2vec_model is None:
                print("Modello Word2Vec non trovato, tentativo di caricamento/addestramento...")

                # Definisce il percorso standard per il modello Word2Vec
                word2vec_path = "models/word2vec_model.bin"

                # Tenta di caricare un modello Word2Vec esistente
                if os.path.exists(word2vec_path):
                    try:
                        from gensim.models import Word2Vec
                        # Carica il modello Word2Vec preaddestrato
                        self.recommender.word2vec_model = Word2Vec.load(word2vec_path)
                        print("Modello Word2Vec caricato da file esistente")
                    except Exception as e:
                        print(f"Errore nel caricamento del modello Word2Vec: {e}")
                        return False
                else:
                    # Se il modello non esiste, deve essere addestrato ex novo
                    print("Addestramento del modello Word2Vec in corso...")
                    if hasattr(self.recommender, 'train_word2vec'):
                        try:
                            # Addestra il modello Word2Vec sui dati disponibili
                            self.recommender.train_word2vec()
                            print("Modello Word2Vec addestrato con successo")

                            # Salva il modello appena addestrato per usi futuri
                            self.recommender.word2vec_model.save(word2vec_path)
                            print(f"Modello salvato in {word2vec_path}")
                        except Exception as e:
                            print(f"Errore nell'addestramento del modello Word2Vec: {e}")
                            return False
                    else:
                        print("Metodo train_word2vec non disponibile nel recommender")
                        return False
            else:
                print("Modello Word2Vec già caricato correttamente")

        return success

    def check_ingredient_coverage(self):
        """
        Verifica quanti degli ingredienti di test esistono nel dataset di ricette.

        Questo metodo analizza la copertura degli ingredienti confrontando gli ingredienti
        nei set di test con tutti gli ingredienti presenti nel dataset delle ricette.
        È utile per identificare ingredienti che potrebbero non essere ben rappresentati
        nel dataset e potrebbero quindi produrre raccomandazioni di scarsa qualità.

        Returns:
            dict: Dizionario contenente i risultati della copertura per ogni set di test.
                 Ogni voce contiene:
                 - total: numero totale di ingredienti nel set di test
                 - found: numero di ingredienti trovati nel dataset
                 - coverage_pct: percentuale di copertura
                 - missing: lista degli ingredienti mancanti dal dataset

        Raises:
            ValueError: Se il recommender non è stato caricato o il DataFrame è None.
        """
        if not self.recommender or self.recommender.df is None:
            raise ValueError("Recommender non caricato. Chiamare load_recommender prima.")

        # Estrae tutti gli ingredienti unici dal dataset
        all_dataset_ingredients = set()

        # Itera attraverso tutte le ricette nel DataFrame
        for ner_list in self.recommender.df['NER']:
            try:
                # Converte la stringa rappresentazione della lista in una lista Python
                # NER (Named Entity Recognition) contiene gli ingredienti estratti
                ingredients = ast.literal_eval(ner_list) if isinstance(ner_list, str) else []

                # Aggiunge tutti gli ingredienti al set, convertendoli in minuscolo
                # per una comparazione case-insensitive
                all_dataset_ingredients.update([ing.lower() for ing in ingredients])
            except:
                # Ignora le righe con dati malformati
                continue

        # Verifica la copertura per ogni set di test
        coverage_results = {}
        for name, ingredients in self.test_ingredients.items():
            # Trova gli ingredienti del test che esistono nel dataset
            found = [ing for ing in ingredients if ing.lower() in all_dataset_ingredients]
            found_count = len(found)
            total_count = len(ingredients)

            # Calcola la percentuale di copertura, gestendo il caso di divisione per zero
            coverage_pct = 0
            if total_count > 0:
                coverage_pct = round(found_count / total_count * 100, 2)

            # Memorizza i risultati della copertura
            coverage_results[name] = {
                'total': total_count,
                'found': found_count,
                'coverage_pct': coverage_pct,
                'missing': [ing for ing in ingredients if ing.lower() not in all_dataset_ingredients]
            }

        return coverage_results

    def check_for_inappropriate_content(self, recipe):
        """
        Verifica se una ricetta contiene contenuti inappropriati.

        Questo metodo analizza il titolo e gli ingredienti di una ricetta per identificare
        la presenza di termini inappropriati che potrebbero rendere la ricetta inadatta
        per certi contesti o utenti.

        Args:
            recipe (dict): Dizionario contenente le informazioni della ricetta.
                          Deve avere almeno le chiavi 'title' e 'ingredients'.

        Returns:
            tuple: Tupla contenente:
                - is_flagged (bool): True se sono stati trovati termini inappropriati
                - matched_terms (list): Lista dei termini inappropriati trovati
        """
        # Combina il titolo e gli ingredienti in un unico testo per l'analisi
        # Converte tutto in minuscolo per una ricerca case-insensitive
        recipe_text = recipe['title'].lower() + ' ' + ' '.join(recipe['ingredients']).lower()

        # Cerca tutti i termini inappropriati nel testo della ricetta
        matched_terms = [term for term in self.inappropriate_terms
                         if term.lower() in recipe_text]

        # Ritorna se la ricetta è stata segnalata e quali termini sono stati trovati
        return (len(matched_terms) > 0, matched_terms)

    def validate_recommendation_relevance(self, ingredients, recipe_ingredients):
        """
        Valida la rilevanza di una ricetta raccomandada rispetto agli ingredienti di input.

        Questo metodo calcola metriche di sovrapposizione tra gli ingredienti forniti
        dall'utente e quelli presenti nella ricetta raccomandata, fornendo una misura
        quantitativa della rilevanza della raccomandazione.

        Args:
            ingredients (list): Lista degli ingredienti forniti dall'utente come query
            recipe_ingredients (list): Lista degli ingredienti presenti nella ricetta raccomandata

        Returns:
            dict: Dizionario contenente le metriche di rilevanza:
                - ingredients_matched: numero di ingredienti in comune
                - ingredients_matched_pct: percentuale degli ingredienti query che sono presenti nella ricetta
                - recipe_coverage_pct: percentuale della ricetta coperta dagli ingredienti query
                - common_ingredients: lista degli ingredienti in comune
        """
        # Converte tutti gli ingredienti in minuscolo per una comparazione uniforme
        query_ingredients = [ing.lower() for ing in ingredients]
        recipe_ingredients_lower = [ing.lower() for ing in recipe_ingredients]

        # Calcola l'intersezione tra i due set di ingredienti
        common_ingredients = set(query_ingredients) & set(recipe_ingredients_lower)
        common_count = len(common_ingredients)

        # Inizializza le percentuali a zero per gestire casi edge
        ingredients_matched_pct = 0
        recipe_coverage_pct = 0

        # Calcola la percentuale di ingredienti query che sono presenti nella ricetta
        if query_ingredients:
            ingredients_matched_pct = round(common_count / len(query_ingredients) * 100, 2)

        # Calcola quanto della ricetta è coperto dagli ingredienti query
        if recipe_ingredients_lower:
            recipe_coverage_pct = round(common_count / len(recipe_ingredients_lower) * 100, 2)

        return {
            'ingredients_matched': common_count,
            'ingredients_matched_pct': ingredients_matched_pct,
            'recipe_coverage_pct': recipe_coverage_pct,
            'common_ingredients': list(common_ingredients)
        }

    def validate_recommendation_coherence(self, recommendations):
        """
        Valida la coerenza delle raccomandazioni multiple.

        Questo metodo analizza se le ricette raccomandate hanno senso insieme,
        utilizzando analisi semantica sui titoli per determinare se appartengono
        a categorie simili o hanno temi comuni.

        Args:
            recommendations (list): Lista di dizionari contenenti le ricette raccomandate

        Returns:
            dict: Dizionario contenente:
                - coherence_score: punteggio di coerenza da 0 a 1
                - analysis: descrizione testuale dell'analisi di coerenza
        """
        if not recommendations:
            return {'coherence_score': 0, 'analysis': "Nessuna raccomandazione da analizzare"}

        # Estrae i titoli delle ricette per l'analisi semantica
        titles = [recipe['title'] for recipe in recommendations]

        # Se c'è solo una ricetta, la coerenza è perfetta per definizione
        if len(titles) < 2:
            return {'coherence_score': 1.0, 'analysis': "Solo una ricetta, coerenza perfetta per definizione"}

        # Utilizza CountVectorizer per creare rappresentazioni vettoriali dei titoli
        # Rimuove le stop words inglesi per migliorare l'analisi semantica
        vectorizer = CountVectorizer(stop_words='english')

        try:
            # Trasforma i titoli in vettori di features (bag of words)
            title_vectors = vectorizer.fit_transform(titles)

            # Calcola la similarità coseno tra tutte le coppie di titoli
            similarities = cosine_similarity(title_vectors)

            # Calcola la similarità media escludendo la diagonale (auto-similarità)
            n = similarities.shape[0]
            total_sim = 0
            if n > 1:  # Evita divisione per zero
                # Somma totale meno la diagonale, diviso per il numero di coppie
                total_sim = (similarities.sum() - n) / (n * (n - 1))

            # Analizza il livello di coerenza basato sul punteggio di similarità
            analysis = ""
            if total_sim < 0.1:
                analysis = "Coerenza molto bassa - le ricette sembrano non correlate"
            elif total_sim < 0.3:
                analysis = "Coerenza bassa - le ricette sono parzialmente correlate"
            elif total_sim < 0.6:
                analysis = "Coerenza moderata - le ricette mostrano un pattern chiaro"
            else:
                analysis = "Coerenza alta - le ricette sono fortemente correlate"

            return {
                'coherence_score': float(total_sim),
                'analysis': analysis
            }
        except Exception as e:
            return {'coherence_score': 0, 'analysis': f"Impossibile calcolare la coerenza: {str(e)}"}

    def validate_single_test_case(self, test_name, ingredients, k=10):
        """
        Valida le raccomandazioni per un singolo caso di test.

        Questo metodo esegue una validazione completa per un set specifico di ingredienti,
        includendo test di rilevanza, contenuto inappropriato, coerenza e prestazioni.

        Args:
            test_name (str): Nome identificativo del caso di test
            ingredients (list): Lista degli ingredienti da testare
            k (int): Numero di raccomandazioni da richiedere

        Returns:
            dict: Dizionario contenente tutti i risultati della validazione per questo test

        Raises:
            ValueError: Se il recommender non è stato caricato
        """
        if not self.recommender:
            raise ValueError("Recommender non caricato")

        print(f"Testing: {test_name} - {ingredients}")

        # Misura il tempo di esecuzione della query
        start_time = time.time()
        results = self.recommender.recommend(ingredients, k=k)
        query_time = time.time() - start_time

        # Inizializza le metriche di validazione
        validated_results = []
        flagged_count = 0
        relevance_scores = []

        # Valida ogni singola raccomandazione
        for recipe in results:
            # Verifica la presenza di contenuti inappropriati
            is_flagged, matched_terms = self.check_for_inappropriate_content(recipe)
            if is_flagged:
                flagged_count += 1

            # Calcola la rilevanza della raccomandazione
            relevance = self.validate_recommendation_relevance(ingredients, recipe['ingredients'])
            relevance_scores.append(relevance['ingredients_matched_pct'])

            # Memorizza i risultati della validazione per questa ricetta
            validated_results.append({
                'recipe': recipe,
                'is_flagged': is_flagged,
                'matched_terms': matched_terms,
                'relevance': relevance
            })

        # Valida la coerenza complessiva delle raccomandazioni
        coherence = self.validate_recommendation_coherence(results)

        # Calcola il punteggio medio di rilevanza, gestendo il caso di lista vuota
        avg_relevance_score = 0
        if relevance_scores:
            avg_relevance_score = sum(relevance_scores) / len(relevance_scores)

        # Prepara il risultato completo della validazione
        validation_result = {
            'test_name': test_name,
            'ingredients': ingredients,
            'query_time': query_time,
            'results_count': len(results),
            'flagged_count': flagged_count,
            'avg_relevance_score': avg_relevance_score,
            'coherence': coherence,
            'validated_results': validated_results
        }

        # Applica filtro di soglia per la coerenza
        # Se la coerenza è troppo bassa, filtra i risultati
        coherence_threshold = 0.5
        if coherence['coherence_score'] < coherence_threshold:
            validation_result['filtered_out'] = True
            validation_result['validated_results'] = []
            validation_result['results_count'] = 0
            validation_result['flagged_count'] = 0
            validation_result['avg_relevance_score'] = 0

        # Aggiunge il risultato alla collezione complessiva
        self.validation_results.append(validation_result)

        return validation_result

    def run_validation(self, test_selection=None, k=10):
        """
        Esegue la validazione su multipli casi di test.

        Questo metodo coordina l'esecuzione di tutti i test di validazione,
        fornendo un'analisi completa delle prestazioni del sistema di raccomandazione.

        Args:
            test_selection (list, optional): Lista dei nomi dei test da eseguire.
                                           Se None, esegue tutti i test disponibili.
            k (int): Numero di raccomandazioni da richiedere per ogni test

        Returns:
            pandas.DataFrame: DataFrame con il riassunto dei risultati di validazione

        Raises:
            ValueError: Se il recommender non è stato caricato
        """
        # Se k è zero o negativo, salta la validazione
        if k <= 0:
            print("Validazione saltata (k=0)")
            return pd.DataFrame()

        if not self.recommender:
            raise ValueError("Recommender non caricato")

        # Seleziona quali test eseguire
        if test_selection:
            # Filtra solo i test specificati che esistono effettivamente
            tests = {name: self.test_ingredients[name]
                     for name in test_selection
                     if name in self.test_ingredients}
        else:
            # Usa tutti i test disponibili
            tests = self.test_ingredients

        if not tests:
            print("Nessun test valido da eseguire")
            return pd.DataFrame()

        # Resetta i risultati precedenti
        self.validation_results = []
        print(f"Esecuzione di {len(tests)} test di validazione...")

        # Esegue ogni test e mostra un riassunto dei risultati
        for name, ingredients in tests.items():
            result = self.validate_single_test_case(name, ingredients, k=k)
            print(f"  {name}: {result['results_count']} risultati, {result['flagged_count']} segnalati, " +
                  f"rilevanza media: {result['avg_relevance_score']:.2f}%, coerenza: {result['coherence']['coherence_score']:.2f}")

        # Crea e ritorna il DataFrame riassuntivo
        return self.create_summary_dataframe()

    def create_summary_dataframe(self):
        """
        Crea un DataFrame riassuntivo dai risultati della validazione.

        Questo metodo trasforma i risultati dettagliati della validazione in un formato
        tabulare facile da leggere e analizzare, adatto per reporting e visualizzazione.

        Returns:
            pandas.DataFrame: DataFrame contenente il riassunto della validazione con colonne:
                - test_name: nome del test
                - ingredients: ingredienti testati
                - results_count: numero di risultati ottenuti
                - query_time: tempo di esecuzione della query
                - flagged_count: numero di risultati segnalati
                - flagged_pct: percentuale di risultati segnalati
                - avg_relevance: rilevanza media in percentuale
                - coherence_score: punteggio di coerenza
                - coherence_analysis: analisi testuale della coerenza
        """
        if not self.validation_results:
            return pd.DataFrame()

        summary = []
        for result in self.validation_results:
            # Calcola la percentuale di risultati segnalati, gestendo la divisione per zero
            flagged_pct = "N/A"
            if result['results_count'] > 0:
                flagged_pct = f"{(result['flagged_count'] / result['results_count'] * 100):.1f}%"

            # Crea una riga del riassunto con tutte le metriche formattate
            summary.append({
                'test_name': result['test_name'],
                'ingredients': ', '.join(result['ingredients']),
                'results_count': result['results_count'],
                'query_time': f"{result['query_time']:.4f}s",
                'flagged_count': result['flagged_count'],
                'flagged_pct': flagged_pct,
                'avg_relevance': f"{result['avg_relevance_score']:.1f}%",
                'coherence_score': f"{result['coherence']['coherence_score']:.2f}",
                'coherence_analysis': result['coherence']['analysis']
            })

        return pd.DataFrame(summary)

    def save_validation_results(self, output_dir="validation_results"):
        """
        Salva i risultati dettagliati della validazione su file.

        Questo metodo persiste i risultati della validazione sia in formato JSON dettagliato
        che in formato CSV riassuntivo per facilitare analisi successive e reporting.

        Args:
            output_dir (str): Directory dove salvare i risultati. Viene creata se non esiste.
        """
        if not self.validation_results:
            print("Nessun risultato di validazione da salvare")
            return

        # Crea la directory di output se non esiste
        os.makedirs(output_dir, exist_ok=True)

        # Salva i risultati dettagliati in formato JSON
        detailed_path = os.path.join(output_dir, "detailed_validation_results.json")
        with open(detailed_path, 'w') as f:
            # Converte i valori numpy in tipi Python nativi per la serializzazione JSON
            results_copy = []
            for result in self.validation_results:
                result_copy = result.copy()
                # Gestisce specificamente i valori numpy che potrebbero causare errori di serializzazione
                if 'avg_relevance_score' in result_copy and isinstance(result_copy['avg_relevance_score'], np.ndarray):
                    result_copy['avg_relevance_score'] = float(result_copy['avg_relevance_score'])
                results_copy.append(result_copy)

            # Salva con indentazione per migliorare la leggibilità
            json.dump(results_copy, f, indent=2, default=str)

        # Crea e salva il riassunto in formato CSV
        summary_df = self.create_summary_dataframe()
        summary_path = os.path.join(output_dir, "validation_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"Risultati dettagliati salvati in {detailed_path}")
        print(f"Riassunto salvato in {summary_path}")

    def analyze_validation_results(self):
        """
        Analizza i risultati della validazione per identificare pattern e problemi.

        Questo metodo esegue un'analisi aggregata di tutti i risultati di validazione
        per fornire insights complessivi sulle prestazioni del sistema di raccomandazione,
        identificando punti di forza e aree di miglioramento.

        Returns:
            dict: Dizionario contenente l'analisi completa con:
                - test_count: numero totale di test eseguiti
                - avg_query_time: tempo medio di esecuzione delle query
                - avg_relevance_score: punteggio medio di rilevanza
                - avg_coherence_score: punteggio medio di coerenza
                - flagged_content: analisi del contenuto inappropriato
                - best_performing: test con le migliori prestazioni
                - worst_performing: test con le peggiori prestazioni
        """
        if not self.validation_results:
            return {"error": "Nessun risultato di validazione da analizzare"}

        # Raccoglie metriche aggregate da tutti i test
        all_flagged_terms = []
        all_relevance_scores = []
        query_times = []
        coherence_scores = []

        # Itera attraverso tutti i risultati per estrarre le metriche
        for result in self.validation_results:
            query_times.append(result['query_time'])
            coherence_scores.append(result['coherence']['coherence_score'])

            # Raccoglie tutti i termini inappropriati trovati
            for validated in result['validated_results']:
                if validated['is_flagged']:
                    all_flagged_terms.extend(validated['matched_terms'])

            # Raccoglie tutti i punteggi di rilevanza
            for validated in result['validated_results']:
                all_relevance_scores.append(validated['relevance']['ingredients_matched_pct'])

        # Analizza la frequenza dei termini inappropriati
        flagged_term_counts = Counter(all_flagged_terms)
        most_common_flagged = flagged_term_counts.most_common(5)

        # Calcola le metriche medie, gestendo il caso di liste vuote
        avg_query_time = 0
        avg_relevance_score = 0
        avg_coherence_score = 0

        if query_times:
            avg_query_time = sum(query_times) / len(query_times)
        if all_relevance_scores:
            avg_relevance_score = sum(all_relevance_scores) / len(all_relevance_scores)
        if coherence_scores:
            avg_coherence_score = sum(coherence_scores) / len(coherence_scores)

        # Prepara le statistiche complessive
        analysis = {
            'test_count': len(self.validation_results),
            'avg_query_time': avg_query_time,
            'avg_relevance_score': avg_relevance_score,
            'avg_coherence_score': avg_coherence_score,
            'flagged_content': {
                'total_flags': sum(flagged_term_counts.values()),
                'unique_terms': len(flagged_term_counts),
                'most_common': most_common_flagged
            }
        }

        # Identifica i test con le migliori e peggiori prestazioni
        if self.validation_results:
            # Ordina i risultati per punteggio di rilevanza medio
            sorted_results = sorted(self.validation_results,
                                    key=lambda x: x['avg_relevance_score'],
                                    reverse=True)

            # Identifica il test con le migliori prestazioni
            analysis['best_performing'] = {
                'test_name': sorted_results[0]['test_name'],
                'ingredients': sorted_results[0]['ingredients'],
                'relevance_score': sorted_results[0]['avg_relevance_score']
            }

            # Identifica il test con le peggiori prestazioni
            analysis['worst_performing'] = {
                'test_name': sorted_results[-1]['test_name'],
                'ingredients': sorted_results[-1]['ingredients'],
                'relevance_score': sorted_results[-1]['avg_relevance_score']
            }

        return analysis


def main():
    """
    Funzione principale che coordina l'esecuzione del sistema di validazione.

    Questa funzione:
    1. Inizializza il sistema di validazione
    2. Carica il modello di raccomandazione
    3. Esegue tutti i test di validazione
    4. Genera e salva i report dei risultati
    5. Fornisce un'analisi aggregata delle prestazioni

    La funzione gestisce tutti gli aspetti del processo di validazione,
    dalla configurazione iniziale alla generazione dei report finali.
    """
    # Parametri di configurazione per l'esecuzione dei test
    k = 10  # Numero di raccomandazioni da richiedere per ogni test
    test_selection = None  # Se None, esegue tutti i test disponibili

    print("Sistema di Validazione Raccomandazioni Ricette")
    print("=======================================")

    # Inizializza il validatore con i file di configurazione
    # Il validatore carica automaticamente gli ingredienti di test e i termini inappropriati
    validator = RecipeValidator(
        ingredients_file="validation/test_ingredients.json",  # File JSON con i set di ingredienti per i test
        inappropriate_terms_file="validation/inappropriate_terms.txt"  # File con termini da filtrare
    )

    # Carica il sistema di raccomandazione con i modelli preaddestrati
    print("Caricamento del modello recommender...")
    model_loaded = validator.load_recommender(
        "models/recipes_faiss.index",  # Indice FAISS per la ricerca vettoriale veloce
        "models/recipes_dataframe.csv"  # DataFrame con i dati delle ricette
    )

    # Verifica che il caricamento del modello sia andato a buon fine
    if not model_loaded:
        print("Fallimento nel caricamento del modello recommender.")
        return

    # Verifica che il modello Word2Vec sia disponibile
    # Questo controllo aggiuntivo assicura che tutti i componenti necessari siano pronti
    print("Verifica del modello Word2Vec...")
    if hasattr(validator, 'train_word2vec_if_needed'):
        word2vec_ready = validator.train_word2vec_if_needed()
        if not word2vec_ready:
            print("Impossibile preparare il modello Word2Vec.")
            return

    print("Tutti i modelli sono pronti")

    # Esegue la suite completa di test di validazione
    try:
        # Esegue tutti i test e ottiene un DataFrame riassuntivo
        summary_df = validator.run_validation(test_selection=test_selection, k=k)

        # Mostra il riassunto dei risultati nella console
        print("\nRiassunto Validazione:")
        print(summary_df)

        # Salva tutti i risultati su file per analisi future
        validator.save_validation_results()

        # Esegue un'analisi aggregata dei risultati
        analysis = validator.analyze_validation_results()
        print("\nAnalisi Validazione:")
        print(json.dumps(analysis, indent=2))

    except Exception as e:
        print(f"Errore durante la validazione: {e}")
        return

    # Ritorna l'istanza del validatore per permettere analisi interattive aggiuntive
    return validator


# Punto di ingresso del programma
# Esegue la funzione main solo se il file viene eseguito direttamente
# (non quando viene importato come modulo)
if __name__ == "__main__":
    main()