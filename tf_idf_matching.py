import psycopg2
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', force=True)


#############################
#  OPTIONAL: Synonyms
#############################
SYNONYM_MAP = {
    "bell": "sweet",
    "courgette": "zucchini",
    # add more synonyms or expansions...
}

def expand_synonyms(text):
    """
    Tokenize and expand synonyms inline.
    e.g. "bell pepper" => "bell sweet pepper"
    """
    tokens = text.lower().split()
    expanded = []
    for t in tokens:
        if t in SYNONYM_MAP:
            expanded.append(t)
            expanded.append(SYNONYM_MAP[t])
        else:
            expanded.append(t)
    return " ".join(expanded)


#############################
#  DB or CSV LOADING
#############################
def load_training_data_from_db():
    """
    Example: load your labeled data from a table 
      `local_usda_mappings(local_ing_text TEXT, usda_id INT)`
    Return as a pandas DataFrame with columns: ['local_text', 'label_id'].
    """
    conn = psycopg2.connect(
        host="172.26.192.1",
        database="mealplanning",
        user="postgres",
        password="new-website-app",
        port="5432"
    )
    query = "SELECT local_ing_text, usda_id FROM local_usda_mappings;"
    df = pd.read_sql(query, conn)
    conn.close()

    df.columns = ["local_text", "label_id"]  # rename for clarity
    return df


#############################
#  MAIN: TRAIN TF-IDF + LOGREG
#############################
def train_tfidf_model():
    """
    1) Load data (local_text -> usda_id).
    2) Preprocess with synonyms.
    3) Train TF-IDF + LogisticRegression pipeline.
    4) Evaluate and save the model.
    """
    logging.info("Loading training data from DB...")
    df = load_training_data_from_db()
    
    # Expand synonyms or do other cleaning
    df["local_text"] = df["local_text"].apply(expand_synonyms)

    # E.g. We want to treat usda_id as a classification label
    # If you have thousands of unique IDs, you might consider a multi-class approach
    # or a top-k approach. For demonstration, we'll do standard classification.

    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Unique USDA labels: {df['label_id'].nunique()}")

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])
    logging.info(f"Train size: {train_df.shape}, Test size: {test_df.shape}")

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            # You can tune parameters here
            # e.g. ngram_range=(1,2), min_df=2, max_df=0.8
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            # You can tune C=, penalty=, etc.
        ))
    ])

    # Train
    pipeline.fit(train_df["local_text"], train_df["label_id"])

    # Evaluate
    preds = pipeline.predict(test_df["local_text"])
    logging.info("\n" + classification_report(test_df["label_id"], preds))

    # Save model for later
    with open("tfidf_logreg.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    logging.info("Model saved to tfidf_logreg.pkl")


#############################
#  DEMO: PREDICT NEW INGREDIENT
#############################
def predict_new_ingredient(ingredient_str, model_path="tfidf_logreg.pkl"):
    # Load pipeline
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Preprocess (expand synonyms, etc.)
    processed = expand_synonyms(ingredient_str)
    predicted_label = pipeline.predict([processed])[0]
    # If you want top k predictions:
    # predicted_probs = pipeline.predict_proba([processed])[0]
    # top_k_indices = predicted_probs.argsort()[::-1][:5]  # top 5
    # top_k_labels = pipeline.named_steps["clf"].classes_[top_k_indices]

    return predicted_label


if __name__ == "__main__":
    train_tfidf_model()

    # Example usage
    test_ing = "green bell pepper"
    pred_label = predict_new_ingredient(test_ing)
    logging.info(f"'{test_ing}' => predicted USDA ID: {pred_label}")
