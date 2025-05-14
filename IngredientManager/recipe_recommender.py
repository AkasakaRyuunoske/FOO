import json
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import ast
import os
import time
import numpy as np


class RecipeRecommender:
    """A recommendation system for recipes based on ingredients using FAISS and SentenceTransformer."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the recommender with a sentence transformer model."""
        print(f"Initializing RecipeRecommender with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Model initialized successfully")
        self.index = None
        self.df = None

    def load_dataset(self, file_paths, output_path=None):
        """Load and prepare datasets from multiple files."""
        print("\n=== LOADING DATASET ===")

        # Convert single path to list
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # Load each file
        dfs = []
        total_recipes = 0

        print(f"Loading {len(file_paths)} dataset file(s)...")
        for i, file_path in enumerate(file_paths):
            print(f"[{i + 1}/{len(file_paths)}] Loading {os.path.basename(file_path)}...")
            try:
                start_time = time.time()
                df = pd.read_csv(file_path)
                load_time = time.time() - start_time

                # Keep only necessary columns
                if all(col in df.columns for col in ["title", "NER", "link"]):
                    df = df[["title", "NER", "link"]]
                    before_count = len(df)
                    df.dropna(subset=["NER"], inplace=True)
                    after_count = len(df)

                    dfs.append(df)
                    total_recipes += after_count

                    print(f"  ✓ Loaded {after_count} recipes in {load_time:.2f}s")
                    if before_count != after_count:
                        print(f"    (Removed {before_count - after_count} rows with missing ingredients)")
                else:
                    print(f"  ✗ Required columns not found in {file_path}")
            except Exception as e:
                print(f"  ✗ Error loading {file_path}: {e}")

        if not dfs:
            print("✗ No valid datasets were loaded")
            raise ValueError("No valid datasets were loaded")

        # Combine all dataframes
        print("\nCombining datasets...")
        start_time = time.time()
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"✓ Combined {len(dfs)} files with {len(self.df)} total recipes in {time.time() - start_time:.2f}s")

        # Remove duplicates
        print("\nRemoving duplicate recipes...")
        start_time = time.time()
        before_count = len(self.df)
        self.df.drop_duplicates(subset=["title", "NER"], inplace=True)
        after_count = len(self.df)
        duplicates_removed = before_count - after_count

        print(f"✓ Removed {duplicates_removed} duplicates in {time.time() - start_time:.2f}s")
        print(f"✓ Final dataset contains {after_count} unique recipes")

        # Save combined dataset if requested
        if output_path:
            print(f"\nSaving combined dataset to {output_path}...")
            start_time = time.time()
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            self.df.to_csv(output_path, index=False)
            print(f"✓ Dataset saved in {time.time() - start_time:.2f}s")

        print("=== DATASET LOADING COMPLETE ===\n")
        return self.df

    def _parse_ingredients(self, ner_string):
        """Parse ingredients from NER string."""
        if isinstance(ner_string, str):
            try:
                ingredients_list = ast.literal_eval(ner_string)
                return ' '.join(ingredients_list)
            except (ValueError, SyntaxError):
                return ''
        return ''

    def _create_embeddings(self):
        """Create embeddings for the loaded dataset."""
        print("\n=== CREATING EMBEDDINGS ===")

        if self.df is None:
            print("✗ No dataset loaded")
            raise ValueError("No dataset loaded. Call load_dataset first.")

        # Extract ingredients text
        print("Processing ingredient data...")
        start_time = time.time()
        self.df['ner_text'] = self.df['NER'].apply(self._parse_ingredients)
        print(f"✓ Processed ingredient data in {time.time() - start_time:.2f}s")

        # Filter out empty texts
        print("Filtering out recipes with empty ingredient lists...")
        start_time = time.time()
        before_count = len(self.df)
        self.df = self.df[self.df['ner_text'].str.strip() != '']
        after_count = len(self.df)

        print(
            f"✓ Removed {before_count - after_count} recipes with empty ingredients in {time.time() - start_time:.2f}s")
        print(f"✓ Proceeding with {after_count} valid recipes")

        # Create embeddings
        print(f"\nGenerating embeddings for {len(self.df)} recipes...")
        print("(This is typically the most time-consuming step. Please wait...)")
        start_time = time.time()

        embeddings = self.model.encode(
            self.df['ner_text'].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True
        )

        total_time = time.time() - start_time
        recipes_per_second = len(self.df) / total_time if total_time > 0 else 0

        print(f"✓ Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        print(f"✓ Embedding completed in {total_time:.2f}s ({recipes_per_second:.1f} recipes/second)")
        print("=== EMBEDDING CREATION COMPLETE ===\n")

        return embeddings

    def build_index(self, embeddings=None, index_path=None, df_path=None):
        """Build FAISS index from embeddings."""
        print("\n=== BUILDING SEARCH INDEX ===")

        # Create embeddings if not provided
        if embeddings is None:
            print("No pre-computed embeddings provided, creating embeddings...")
            embeddings = self._create_embeddings()
        else:
            print(f"Using provided embeddings with shape {embeddings.shape}")

        # Create FAISS index
        print("\nInitializing FAISS index...")
        start_time = time.time()
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        print(f"✓ Created IndexFlatL2 with dimension {dimension} in {time.time() - start_time:.2f}s")

        # Add embeddings to index
        print(f"Adding {len(embeddings)} vectors to index...")
        start_time = time.time()
        self.index.add(embeddings)
        add_time = time.time() - start_time
        vectors_per_second = len(embeddings) / add_time if add_time > 0 else 0

        print(f"✓ Added vectors in {add_time:.2f}s ({vectors_per_second:.1f} vectors/second)")
        print(f"✓ Index now contains {self.index.ntotal} vectors")

        # Save index if requested
        if index_path:
            print(f"\nSaving FAISS index to {index_path}...")
            start_time = time.time()
            os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else '.', exist_ok=True)
            faiss.write_index(self.index, index_path)
            print(f"✓ Index saved in {time.time() - start_time:.2f}s")

        # Save DataFrame if requested
        if df_path and self.df is not None:
            print(f"Saving DataFrame to {df_path}...")
            start_time = time.time()
            os.makedirs(os.path.dirname(df_path) if os.path.dirname(df_path) else '.', exist_ok=True)
            self.df.to_csv(df_path, index=False)
            print(f"✓ DataFrame saved in {time.time() - start_time:.2f}s")

        print("=== INDEX BUILDING COMPLETE ===\n")
        return self.index

    def load_model(self, index_path, df_path):
        """Load existing FAISS index and DataFrame."""
        print("\n=== LOADING MODEL ===")
        try:
            print(f"Loading FAISS index from {index_path}...")
            start_time = time.time()
            self.index = faiss.read_index(index_path)
            index_load_time = time.time() - start_time
            print(f"✓ Index loaded in {index_load_time:.2f}s with {self.index.ntotal} vectors")

            print(f"Loading DataFrame from {df_path}...")
            start_time = time.time()
            self.df = pd.read_csv(df_path)
            df_load_time = time.time() - start_time
            print(f"✓ DataFrame loaded in {df_load_time:.2f}s with {len(self.df)} recipes")

            print("=== MODEL LOADING COMPLETE ===\n")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("=== MODEL LOADING FAILED ===\n")
            return False

    def _get_ingredients_list(self, ner_string):
        """Convert NER string to a list of ingredients."""
        if isinstance(ner_string, str):
            try:
                return ast.literal_eval(ner_string)
            except (ValueError, SyntaxError):
                return []
        return []

    def recommend(self, ingredients, k=10):
        """Recommend recipes based on ingredients."""
        print("\n=== SEARCHING FOR RECIPES ===")

        if self.index is None or self.df is None:
            print("✗ Model not built or loaded yet")
            raise ValueError("Model not built or loaded yet")

        # Join ingredients and create embedding
        print(f"Searching with ingredients: {', '.join(ingredients)}")

        print("Converting ingredients to embedding...")
        start_time = time.time()
        query_text = ' '.join(ingredients)
        query_vector = self.model.encode([query_text], convert_to_numpy=True)
        print(f"✓ Created query embedding in {time.time() - start_time:.4f}s")

        # Search top k matches
        k = min(k, len(self.df))
        print(f"Searching for top {k} matches among {self.index.ntotal} recipes...")
        start_time = time.time()
        distances, indices = self.index.search(query_vector, k=k)
        search_time = time.time() - start_time

        print(f"✓ Search completed in {search_time:.4f}s")

        # Format results
        print("Formatting results...")
        start_time = time.time()
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.df):  # Ensure index is valid
                row = self.df.iloc[idx]
                ingredients_list = self._get_ingredients_list(row['NER'])

                results.append({
                    'score': float(distances[0][i]),
                    'title': row['title'],
                    'ingredients': ingredients_list,
                    'link': row['link'] if 'link' in self.df.columns else 'No link available'
                })

        print(f"✓ Formatted {len(results)} results in {time.time() - start_time:.4f}s")
        print("=== SEARCH COMPLETE ===\n")
        return results


def test_recommender(recommender, test_cases_file=None):
    """Test the recommender with sample ingredient combinations."""
    print("\n=== TESTING RECOMMENDER ===")

    if test_cases_file and os.path.exists(test_cases_file):
        with open(test_cases_file, 'r') as f:
            test_cases_dict = json.load(f)
        test_cases = list(test_cases_dict.values())
        print(f"✓ Loaded {len(test_cases)} test ingredient sets from {test_cases_file}")
    else:
        print(f"Test Ingredients file not found: {test_cases_file}")

    for i, ingredients in enumerate(test_cases):
        print(f"\nTest {i + 1}/{len(test_cases)}: {', '.join(ingredients)}")

        start_time = time.time()
        results = recommender.recommend(ingredients, k=5)
        elapsed = time.time() - start_time

        print(f"Found {len(results)} matches in {elapsed:.4f} seconds")

        for j, recipe in enumerate(results):
            print(f"{j + 1}. {recipe['title']}")
            print(f"   Similarity score: {recipe['score']:.4f}")
            print(f"   Ingredients: {', '.join(recipe['ingredients'])}")
            print(f"   Link: {recipe['link']}")
        print("-" * 50)

    print("=== TESTING COMPLETE ===")


def main():
    print("\n====== RECIPE RECOMMENDER SYSTEM ======\n")

    os.makedirs("models", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)

    dataset_path = "dataset"
    index_path = "models/recipes_faiss.index"
    df_path = "models/recipes_dataframe.csv"
    test_cases_file = "validation/test_ingredients.json"

    chunks_to_use = [2]  # All desired chunks
    file_paths = [f"{dataset_path}/recipies_dataset_tagged_chunk_{size}%.csv" for size in chunks_to_use]
    file_paths = [f for f in file_paths if os.path.exists(f)]

    recommender = RecipeRecommender()

    if os.path.exists(index_path) and os.path.exists(df_path):
        print("Loading existing model...")
        if recommender.load_model(index_path, df_path):
            print("✓ Model loaded.")

            # Load new chunks only (filter out already loaded recipes by title+NER)
            loaded_titles = set(zip(recommender.df['title'], recommender.df['NER']))
            new_dfs = []

            for path in file_paths:
                df_new = pd.read_csv(path)
                df_new = df_new[["title", "NER", "link"]].dropna(subset=["NER"])
                df_new["ner_text"] = df_new["NER"].apply(recommender._parse_ingredients)
                df_new = df_new[df_new["ner_text"].str.strip() != '']
                df_new = df_new[~df_new.apply(lambda row: (row['title'], row['NER']) in loaded_titles, axis=1)]

                if not df_new.empty:
                    print(f"✓ Adding {len(df_new)} new recipes from {path}")
                    new_embeddings = recommender.model.encode(df_new["ner_text"].tolist(), convert_to_numpy=True)
                    recommender.index.add(new_embeddings)
                    recommender.df = pd.concat([recommender.df, df_new], ignore_index=True)

            # Save updated model
            faiss.write_index(recommender.index, index_path)
            recommender.df.to_csv(df_path, index=False)

            test_recommender(recommender, test_cases_file=test_cases_file)
        else:
            print("✗ Failed to load model.")
    else:
        print("No existing model. Building from scratch...")
        df = recommender.load_dataset(file_paths)
        recommender.build_index(index_path=index_path, df_path=df_path)
        test_recommender(recommender, test_cases_file=test_cases_file)

    print("\n====== RECIPE RECOMMENDER COMPLETE ======")


if __name__ == "__main__":
    main()