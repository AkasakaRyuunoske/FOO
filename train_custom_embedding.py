import glob

import nltk
import numpy as np
import pandas
import pandas as pd
from gensim.models import Word2Vec

nltk.download("punkt_tab")
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib


# Average word embeddings for each recipe
def avg_embedding(tokens, model):
    embeddings = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)


if __name__ == "__main__":
    print("Reading dataset...")

        # Uncomment to use whole dataset (all parts)
    # # Load and concatenate multiple CSV files
    # files = glob.glob("dataset/recipies_dataset_tagged_chunk_*")
    #
    # # Combine datasets into one DataFrame
    # df_list = [pd.read_csv(f) for f in files]
    # df = pd.concat(df_list, ignore_index=True)
    #
    # print(f"Combined dataset size: {df.shape}")
    df = pd.read_csv("dataset/recipies_dataset_tagged_chunk_10%")
    df["text"] = (df["directions"] + " " + df["ingredients"]).str.lower()

    # Tokenize sentences into words
    df['tokens'] = df['text'].apply(nltk.word_tokenize)

    # Train your own Word2Vec embeddings
    w2v_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=2)

    df['embedding'] = df['tokens'].apply(lambda x: avg_embedding(x, w2v_model))

    # Embeddings as features
    X = np.vstack(df['embedding'].values)
    y_difficulty = df['difficulty']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_difficulty)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # XGBoost Classifier
    clf = xgb.XGBClassifier(
        objective='multi:softmax',  # for binary use 'binary:logistic'
        num_class=len(np.unique(y_encoded)),  # remove this line for binary
        eval_metric='mlogloss',  # 'logloss' for binary
        use_label_encoder=False
    )

    clf.fit(X_train, y_train)

    # Predict & Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Difficulty model
    joblib.dump(clf, 'models/difficulty_classifier/xgb_difficulty_model.joblib')
    joblib.dump(le, 'models/difficulty_classifier/difficulty_label_encoder.joblib')

    w2v_model.save("models/w2v/recipe_word2vec.model")

    # # Cost model
    # joblib.dump(clf_cost, 'xgb_cost_model.joblib')
    # joblib.dump(le_cost, 'cost_label_encoder.joblib')
    #
    # # Time model
    # joblib.dump(clf_time, 'xgb_time_model.joblib')
    # joblib.dump(le_time, 'time_label_encoder.joblib')