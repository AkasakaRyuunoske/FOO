import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import ast  # Safer than eval


def prepare_dataset():
    # Load CSV (adjust file name if needed)
    df = pd.read_csv('dataset/recipies_dataset_tagged_chunk_2%')

    # Keep only necessary columns
    df = df[["title", "NER", "link"]]

    # Drop duplicates and missing values
    df.dropna(subset=["NER"], inplace=True)
    df.drop_duplicates(subset=["NER"], inplace=True)

    # Save cleaned dataset
    df.to_csv("clean_dataset/recipies_dataset_tagged_chunk_2%.csv", index=False)

    return df


def create_embeddings(data_frame):
    # Load pretrained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Prepare ner_text: join ingredients into a single string
    data_frame['ner_text'] = data_frame['NER'].apply(lambda x: ' '.join(ast.literal_eval(x)))

    # Create embeddings
    embeddings = model.encode(data_frame['ner_text'].tolist(), show_progress_bar=True, convert_to_numpy=True)

    return embeddings, model


def build_FAISS(embeddings, df):
    # Get embedding dimensionality
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings
    index.add(embeddings)

    # Save index
    faiss.write_index(index, "recipes_faiss.index")

    # Save DataFrame correctly as CSV
    df.to_csv("recipes_dataframe.csv", index=False)


def test_this_stuff(model):
    # Load FAISS index
    index = faiss.read_index("recipes_faiss.index")

    # Load DataFrame (now from CSV!)
    df = pd.read_csv("recipes_dataframe.csv")

    # User input
    user_ingredients = ["milk", "sugar", "butter"]

    # Join user ingredients
    user_query_text = ' '.join(user_ingredients)

    # Embed user input
    query_vector = model.encode([user_query_text], convert_to_numpy=True)

    # Search top 10 matches
    D, I = index.search(query_vector, k=10)

    # Show results
    for idx in I[0]:
        print(f"Recipe: {df.iloc[idx]['title']}")
        print(f"Ingredients: {df.iloc[idx]['NER']}")
        print(f"Link: {df.iloc[idx]['link'] if 'link' in df.columns else 'No link available'}")
        print("---")


if __name__ == "__main__":
    """
    Dataset (NER) 
        ↓
    Preprocess (clean, join)
        ↓
    Embed (Sentence Transformers)
        ↓
    FAISS Index
        ↓
    User ingredients → Embed → Search → Top recipes
    """
    print("Starting FAISS pipeline...")

    print("Step: Preprocess (clean, join)")
    df = prepare_dataset()

    print("Step: Embed (Sentence Transformers)")
    embeddings, model = create_embeddings(df)

    print("Step: FAISS Index")
    build_FAISS(embeddings, df)

    print("Step: Testing with input => [milk, sugar, butter]")
    test_this_stuff(model)
