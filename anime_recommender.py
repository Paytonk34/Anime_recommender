import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier


# Load anime information
anime = pd.read_csv("anime.csv")
anime = anime.dropna(subset=["name"]).copy()
anime["genre"] = anime["genre"].fillna("")
anime["rating"] = anime["rating"].fillna(0)
anime = anime.drop_duplicates(subset=["anime_id"])
anime = anime.reset_index(drop=True)

# Makes each genre separate to allow user to select one genre at a time
available_genres = sorted({
    genre.strip()
    for entry in anime["genre"]
    for genre in entry.split(",")
    if genre.strip()
})


# Build the content-based genre model
def split_genres(text):
    return [
        genre.strip().lower()
        for genre in text.split(",")
        if genre.strip()
    ]


# Converts test into numerical features
vectorizer = TfidfVectorizer(
    tokenizer=split_genres,
    token_pattern=None,
    lowercase=False
)

genre_matrix = vectorizer.fit_transform(anime["genre"])

# Load historical user ratings
ratings = pd.read_csv(
    "rating.csv",
    dtype={
        "user_id": "int32",
        "anime_id": "int32",
        "rating": "int16"
    }
)

# Ratings of 7 or higher indicate positive feedback
ratings = ratings[ratings["rating"] != -1].copy()
ratings = ratings[ratings["anime_id"].isin(anime["anime_id"])]
ratings = ratings.drop_duplicates(subset=["user_id", "anime_id"])

ratings["liked"] = (ratings["rating"] >= 7).astype(int)

positive_ratings = ratings[ratings["liked"] == 1].copy()


# Genre features for every anime
anime_genres = anime["genre"].apply(
    lambda text: [
        genre.strip()
        for genre in text.split(",")
        if genre.strip()
    ]
)

# Creates binary genre features
mlb = MultiLabelBinarizer()
genre_features = pd.DataFrame(
    mlb.fit_transform(anime_genres),
    columns=mlb.classes_,
    index=anime.index
)

# Create anime feature table
anime_features = genre_features.copy()
anime_features["anime_id"] = anime["anime_id"]
anime_features["members"] = np.log1p(
    anime["members"].fillna(0).clip(lower=0)
)

# Trains the model
training_data = ratings[
    ["anime_id", "liked"]
].merge(
    anime_features,
    on="anime_id",
    how="inner"
)

feature_columns = [
    column
    for column in anime_features.columns
    if column != "anime_id"
]

training_sample = training_data.sample(
    n=min(100000, len(training_data)),
    random_state=42
)

X = training_sample[feature_columns]
y = training_sample["liked"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
anime_probabilities = model.predict_proba(
    anime_features[feature_columns]
)[:, 1]

print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("Baseline Classification Report:")
print(classification_report(
    y_test,
    baseline_predictions,
    zero_division=0
))

# Convert user IDs into consecutive matrix positions
user_codes, unique_users = pd.factorize(positive_ratings["user_id"])

# Match anime IDs to their positions in the anime DataFrame
anime_positions = pd.Series(np.arange(len(anime)), index=anime["anime_id"])

item_codes = positive_ratings["anime_id"].map(anime_positions).to_numpy()

# Build a sparse user-anime interaction matrix to show which users gave positive ratings to which anime
interaction_matrix = coo_matrix(
    (
        np.ones(len(positive_ratings), dtype=np.float32), (user_codes, item_codes)
    ),
    shape=(len(unique_users), len(anime)),
    dtype=np.float32
).tocsr()

# Changes matrix so that each anime is represented by a vector of users who gave a positive rating
item_user_matrix = normalize( interaction_matrix.T.tocsr(), norm="l2", axis=1)

# Count positive interactions for each anime
positive_counts = np.asarray(interaction_matrix.sum(axis=0)).ravel()


# Function that generates recommendations by selected genre
def recommend_by_genres(selected_genre, number_of_recommendations=5):
    if not selected_genre:
        return pd.DataFrame()

    # Calculate genre similarity
    genre_query = ", ".join(selected_genre)
    query_vector = vectorizer.transform([genre_query])

    genre_scores = cosine_similarity(query_vector, genre_matrix).flatten()

    # Only consider anime with matching genres
    eligible = np.where(genre_scores > 0)[0]

    if len(eligible) == 0:
        return pd.DataFrame()

    # Select popular genre-matching anime as reference items
    seed_scores = (genre_scores * np.log1p(positive_counts))

    seed_indices = np.argsort(seed_scores)[::-1][:15]
    seed_indices = seed_indices[seed_scores[seed_indices] > 0]

    if len(seed_indices) == 0:
        return pd.DataFrame()

    # Weight references anime by genre similarity
    seed_weights = genre_scores[seed_indices]
    seed_weights = seed_weights / seed_weights.sum()

    probabilities = anime_probabilities

    # Collaborative filtering:
    # Find anime liked by users who liked the reference anime
    user_profile = (item_user_matrix[seed_indices].T @ seed_weights)
    collaborative_scores = np.asarray(item_user_matrix @ user_profile).ravel()

    # Combine collaborative patterns with genre relevance
    final_scores = (0.3 * collaborative_scores + 0.2 * genre_scores + 0.5 * probabilities)

    # Exclude nonmatching anime and reference items
    final_scores[genre_scores <= 0] = -np.inf
    final_scores[seed_indices] = -np.inf

    # Only recommend anime with collaborative evidence
    final_scores[positive_counts == 0] = -np.inf
    final_scores[collaborative_scores <= 0] = -np.inf

    ranked_indices = np.argsort(final_scores)[::-1]
    ranked_indices = [
        i for i in ranked_indices
        if np.isfinite(final_scores[i])
    ][:number_of_recommendations]

    results = anime.iloc[ranked_indices].copy()

    results["similarity"] = genre_scores[ranked_indices]
    results["predicted like score"] = probabilities[ranked_indices]
    results["recommendation score"] = (final_scores[ranked_indices])

    return results [
        [
            "name",
            "genre",
            "rating",
            "similarity",
            "predicted like score",
            "recommendation score"
        ]
    ].reset_index(drop=True)


if __name__ == "__main__":
    results = recommend_by_genres(["Action", "Fantasy"])
    print(results.to_string(index=False))


