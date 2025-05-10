import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import ast  # Safer than eval
import os
import time
import numpy as np
from pathlib import Path


class RecipeRecommender:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the recommender with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.df = None

    def prepare_dataset(self, file_paths, output_path=None):
        """
        Load and prepare datasets from multiple files.

        Args:
            file_paths: List of file paths or a single file path
            output_path: Optional path to save the combined dataset
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        dfs = []
        for file_path in file_paths:
            print(f"Loading {file_path}...")
            try:
                df = pd.read_csv(file_path)
                # Keep only necessary columns
                if all(col in df.columns for col in ["title", "NER", "link"]):
                    df = df[["title", "NER", "link"]]
                    # Drop duplicates and missing values
                    df.dropna(subset=["NER"], inplace=True)
                    dfs.append(df)
                else:
                    print(f"Warning: Required columns not found in {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if not dfs:
            raise ValueError("No valid datasets were loaded")

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.drop_duplicates(subset=["title", "NER"], inplace=True)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            combined_df.to_csv(output_path, index=False)
            print(f"Combined dataset saved to {output_path}")

        self.df = combined_df
        return combined_df

    def create_embeddings(self):
        """Create and return embeddings for the loaded dataset."""
        if self.df is None:
            raise ValueError("No dataset loaded. Call prepare_dataset first.")

        # Prepare text from NER data
        self.df['ner_text'] = self.df['NER'].apply(lambda x:
                                                   ' '.join(ast.literal_eval(x))
                                                   if isinstance(x, str)
                                                   else '')

        # Filter out any empty texts
        self.df = self.df[self.df['ner_text'].str.strip() != '']

        print(f"Creating embeddings for {len(self.df)} recipes...")
        embeddings = self.model.encode(self.df['ner_text'].tolist(),
                                       show_progress_bar=True,
                                       convert_to_numpy=True)

        return embeddings

    def build_index(self, embeddings=None, index_path=None, df_path=None):
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: Optional pre-computed embeddings
            index_path: Path to save the FAISS index
            df_path: Path to save the DataFrame
        """
        if embeddings is None:
            embeddings = self.create_embeddings()

        # Get embedding dimensionality
        dimension = embeddings.shape[1]

        # Create FAISS index
        self.index = faiss.IndexFlatL2(dimension)

        # Add embeddings
        self.index.add(embeddings)

        if index_path:
            os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else '.', exist_ok=True)
            faiss.write_index(self.index, index_path)
            print(f"FAISS index saved to {index_path}")

        if df_path and self.df is not None:
            os.makedirs(os.path.dirname(df_path) if os.path.dirname(df_path) else '.', exist_ok=True)
            self.df.to_csv(df_path, index=False)
            print(f"DataFrame saved to {df_path}")

        return self.index

    def load_model(self, index_path, df_path):
        """Load existing FAISS index and DataFrame."""
        try:
            self.index = faiss.read_index(index_path)
            self.df = pd.read_csv(df_path)
            print(f"Loaded index with {self.index.ntotal} vectors and DataFrame with {len(self.df)} recipes")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def recommend(self, ingredients, k=10):
        """
        Recommend recipes based on ingredients.

        Args:
            ingredients: List of ingredient strings
            k: Number of recommendations to return

        Returns:
            List of dictionaries with recipe info
        """
        if self.index is None or self.df is None:
            raise ValueError("Model not built or loaded yet")

        # Join ingredients
        query_text = ' '.join(ingredients)

        # Embed query
        query_vector = self.model.encode([query_text], convert_to_numpy=True)

        # Search top k matches
        D, I = self.index.search(query_vector, k=min(k, len(self.df)))

        # Prepare results
        results = []
        for i, idx in enumerate(I[0]):
            if idx < len(self.df):  # Ensure index is valid
                results.append({
                    'score': float(D[0][i]),  # Convert numpy float to Python float
                    'title': self.df.iloc[idx]['title'],
                    'ingredients': ast.literal_eval(self.df.iloc[idx]['NER']) if isinstance(self.df.iloc[idx]['NER'],
                                                                                            str) else [],
                    'link': self.df.iloc[idx]['link'] if 'link' in self.df.columns else 'No link available'
                })

        return results


def test_with_ingredients(recommender, test_cases):
    """
    Test the recommender with multiple ingredient combinations.

    Args:
        recommender: Initialized RecipeRecommender object
        test_cases: List of ingredient lists
    """
    for i, ingredients in enumerate(test_cases):
        print(f"\nTest {i + 1}: Searching with ingredients: {ingredients}")

        start_time = time.time()
        results = recommender.recommend(ingredients, k=5)
        elapsed = time.time() - start_time

        print(f"Found {len(results)} matches in {elapsed:.4f} seconds")

        for j, recipe in enumerate(results):
            print(f"{j + 1}. {recipe['title']}")
            print(f"   Similarity score: {recipe['score']:.4f}")
            print(f"   Ingredients: {recipe['ingredients']}")
            print(f"   Link: {recipe['link']}")

        print("-" * 50)


def evaluate_dataset_sizes(base_path, chunk_sizes, out_dir='evaluation_results'):
    """
    Evaluate performance with different dataset sizes.

    Args:
        base_path: Base path to dataset chunks
        chunk_sizes: List of percentages to test (e.g. [2, 4, 6])
        out_dir: Directory to save results
    """
    os.makedirs(out_dir, exist_ok=True)

    results = []
    recommender = RecipeRecommender()

    # Test cases to use for consistent evaluation
    test_cases = [
        ["milk", "sugar", "butter"],
        ["chicken", "garlic", "onion"],
        ["chocolate", "flour", "eggs"],
        ["rice", "beans", "tomato"]
    ]

    # Individual chunks first
    for size in chunk_sizes:
        print(f"\n{'=' * 20} Testing {size}% chunk {'=' * 20}")
        file_path = f"{base_path}/recipies_dataset_tagged_chunk_{size}%"

        # Skip if file doesn't exist
        if not os.path.exists(file_path):
            print(f"File {file_path} not found, skipping")
            continue

        # Build index
        start_time = time.time()
        df = recommender.prepare_dataset(file_path)
        embeddings = recommender.create_embeddings()
        recommender.build_index(embeddings)
        build_time = time.time() - start_time

        # Get number of recipes
        recipe_count = len(df)

        # Test with sample queries
        query_times = []
        for ingredients in test_cases:
            start = time.time()
            recommender.recommend(ingredients, k=5)
            query_times.append(time.time() - start)

        avg_query_time = np.mean(query_times)
        results.append({
            'type': 'individual',
            'size': size,
            'recipe_count': recipe_count,
            'build_time': build_time,
            'avg_query_time': avg_query_time
        })

    # Now test cumulative chunks
    for i in range(len(chunk_sizes)):
        used_sizes = chunk_sizes[:i + 1]
        total_size = sum(used_sizes)
        print(f"\n{'=' * 20} Testing cumulative {total_size}% ({'+'.join(str(s) for s in used_sizes)}) {'=' * 20}")

        # Collect file paths
        file_paths = [f"{base_path}/recipies_dataset_tagged_chunk_{size}%" for size in used_sizes]
        file_paths = [f for f in file_paths if os.path.exists(f)]

        if not file_paths:
            print("No valid files found, skipping")
            continue

        # Build index
        start_time = time.time()
        df = recommender.prepare_dataset(file_paths)
        embeddings = recommender.create_embeddings()
        recommender.build_index(embeddings)
        build_time = time.time() - start_time

        # Get number of recipes
        recipe_count = len(df)

        # Test with sample queries
        query_times = []
        for ingredients in test_cases:
            start = time.time()
            recommender.recommend(ingredients, k=5)
            query_times.append(time.time() - start)

        avg_query_time = np.mean(query_times)
        results.append({
            'type': 'cumulative',
            'size': total_size,
            'recipe_count': recipe_count,
            'build_time': build_time,
            'avg_query_time': avg_query_time
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_path = f"{out_dir}/dataset_size_evaluation.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation results saved to {results_path}")

    return results_df


def main():
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)

    # Define paths
    dataset_path = "dataset"
    index_path = "models/recipes_faiss.index"
    df_path = "models/recipes_dataframe.csv"

    # Initialize recommender
    recommender = RecipeRecommender()

    # Choose operation mode
    print("FAISS Recipe Recommendation System")
    print("1. Build index from new dataset")
    print("2. Load existing model and test")
    print("3. Evaluate different dataset sizes")
    choice = input("Select an option (1-3): ").strip()

    if choice == '1':
        # Get dataset chunks to use
        available_chunks = sorted([int(f.split('_')[-1].rstrip('%'))
                                   for f in os.listdir(dataset_path)
                                   if 'chunk' in f and f.endswith('%')])

        print(f"Available chunks: {available_chunks}")
        chunks_input = input("Enter chunks to use (comma separated, e.g. '2,4,6' or 'all'): ").strip()

        if chunks_input.lower() == 'all':
            chunks_to_use = available_chunks
        else:
            chunks_to_use = [int(c.strip()) for c in chunks_input.split(',') if c.strip()]

        # Get file paths
        file_paths = [f"{dataset_path}/recipies_dataset_tagged_chunk_{size}%" for size in chunks_to_use]
        file_paths = [f for f in file_paths if os.path.exists(f)]

        if not file_paths:
            print("No valid files found")
            return

        # Prepare, create embeddings, and build index
        recommender.prepare_dataset(file_paths)
        embeddings = recommender.create_embeddings()
        recommender.build_index(embeddings, index_path, df_path)

        # Test with some ingredients
        test_with_ingredients(recommender, [
            ["milk", "sugar", "butter"],
            ["chicken", "garlic", "onion"],
            ["chocolate", "flour", "eggs"]
        ])

    elif choice == '2':
        # Load existing model
        if not os.path.exists(index_path) or not os.path.exists(df_path):
            print(f"Model files not found at {index_path} or {df_path}")
            return

        if recommender.load_model(index_path, df_path):
            # Interactive testing
            print("\nEnter ingredients (comma separated, or 'q' to quit):")
            while True:
                ingredients_input = input("> ").strip()
                if ingredients_input.lower() == 'q':
                    break

                ingredients = [i.strip() for i in ingredients_input.split(',') if i.strip()]
                if not ingredients:
                    continue

                results = recommender.recommend(ingredients)
                print(f"\nFound {len(results)} recipes with {', '.join(ingredients)}:")
                for i, recipe in enumerate(results[:5]):
                    print(f"{i + 1}. {recipe['title']}")
                    print(f"   Similarity score: {recipe['score']:.4f}")
                    print(f"   Ingredients: {recipe['ingredients']}")
                    print(f"   Link: {recipe['link']}")
                    print()

    elif choice == '3':
        # Evaluate different dataset sizes
        available_chunks = sorted([int(f.split('_')[-1].rstrip('%'))
                                   for f in os.listdir(dataset_path)
                                   if 'chunk' in f and f.endswith('%')])

        if not available_chunks:
            print("No dataset chunks found")
            return

        print(f"Available chunks: {available_chunks}")
        chunks_input = input("Enter chunks to evaluate (comma separated, e.g. '2,4,6' or 'all'): ").strip()

        if chunks_input.lower() == 'all':
            chunks_to_evaluate = available_chunks
        else:
            chunks_to_evaluate = [int(c.strip()) for c in chunks_input.split(',') if c.strip()]

        results = evaluate_dataset_sizes(dataset_path, chunks_to_evaluate)

        # Display summary
        print("\nEvaluation Summary:")
        print(results)

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()