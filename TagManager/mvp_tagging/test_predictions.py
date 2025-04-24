import joblib
import nltk
from gensim.models import Word2Vec

from train_custom_embedding import avg_embedding

if __name__ == "__main__":

    # Load trained model and encoder for "difficulty"
    clf_difficulty = joblib.load('../../models/difficulty_classifier/xgb_difficulty_model.joblib')
    le_difficulty = joblib.load('../../models/difficulty_classifier/difficulty_label_encoder.joblib')
    w2v_model = Word2Vec.load('models/w2v/recipe_word2vec.model')

    ingredients = "flour, 3 eggs, 1 tbs sugar"
    directions = "Mix ingredients and bake at 180°C for 20 minutes"
    # Preprocess
    text = (ingredients + " " + directions).lower()
    tokens = nltk.word_tokenize(text.lower())
    embedding = avg_embedding(tokens, w2v_model)

    # Reshape for prediction
    embedding_reshaped = embedding.reshape(1, -1)

    # For new recipe:
    new_text = ingredients + " " + directions
    new_tokens = nltk.word_tokenize(new_text)
    new_embedding = avg_embedding(new_tokens, w2v_model).reshape(1, -1)

    # Make prediction
    predicted_label_encoded = clf_difficulty.predict(new_embedding.reshape(1, -1))
    predicted_label = le_difficulty.inverse_transform(predicted_label_encoded)[0]

    print(f"The predicted difficulty is: {predicted_label}")