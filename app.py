
import os
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

from anime_recommender import anime, recommend_by_genres

FEEDBACK_FILE = "feedback.csv"

# Remember the recommendations being rated
if "current_results" not in st.session_state:
    st.session_state.current_results = None

if "current_title" not in st.session_state:
    st.session_state.current_title = None


available_genres = sorted({
    genre.strip()
    for entry in anime["genre"].dropna()
    for genre in entry.split(",")
    if genre.strip()
})

st.title("Anime Recommendation System")
st.write("Choose your favorite genres to discover anime.")

# Multi-select dropdown
selected_genres = st.multiselect(
    "Choose one or more genres",
    options=available_genres,
    placeholder="Select genres..."
)

if st.button("Get Recommendations"):
    if not selected_genres:
        st.warning("Please select at least one genre.")
    else:
        results = recommend_by_genres(selected_genres)

        st.session_state.current_results = results
        st.session_state.current_title = ", ".join(selected_genres)

# Display recommendations and collect feedback
if st.session_state.current_results is not None:
    results = st.session_state.current_results

    st.subheader("Your Recommendations")
    st.dataframe(results, hide_index=True, width='stretch')

    st.subheader("Rate These Recommendations")

    with st.form("recommendation_feedback"):
        user_rating = st.slider(
            "How would you rate these recommendations overall?",
            min_value=1,
            max_value=10,
            value=8,
            help="1 = Very poor, 10 = Excellent"
        )

        submitted = st.form_submit_button("Submit Rating")

    if submitted:
        new_feedback = pd.DataFrame([{
            "date": datetime.now().isoformat(timespec="seconds"),
            "input_anime": st.session_state.current_title,
            "rating": user_rating,
            "recommended_anime": " | ".join(results["name"].tolist())
        }])

        if os.path.exists(FEEDBACK_FILE):
            new_feedback.to_csv(
                FEEDBACK_FILE,
                mode="a",
                header=False,
                index=False
            )
        else:
            new_feedback.to_csv(
                FEEDBACK_FILE,
                index=False
            )

        st.success("Thank you! Your feedback has been recorded.")

# Show the running average from all saved ratings
if os.path.exists(FEEDBACK_FILE):
    feedback = pd.read_csv(FEEDBACK_FILE)

    if not feedback.empty:
        average_rating = feedback["rating"].mean()
        satisfaction_percentage = average_rating / 10 * 100

        st.divider()
        st.subheader("Recommendation Feedback Summary")

        col1, col2 = st.columns(2)

        col1.metric(
            "Average Satisfaction",
            f"{satisfaction_percentage:.1f}%"
        )

        col2.metric(
            "Ratings Submitted",
            len(feedback)
        )


# Histogram visual of anime ratings
rated_anime = anime[anime["rating"] > 0]

rating_chart = px.histogram(
    rated_anime,
    x="rating",
    nbins=20,
    title="Distribution of Anime Ratings",
    labels={"rating": "Rating", "count": "Number of Anime"}
)
st.plotly_chart(rating_chart, width='stretch')

# Bar graph of top 10 genres
genre_counts = (
    anime["genre"]
    .str.split(",")
    .explode()
    .str.strip()
)

genre_counts = genre_counts[
    genre_counts.notna() & (genre_counts != "")
]

top_genres = genre_counts.value_counts().head(10)
genre_df = top_genres.rename_axis("genre").reset_index(
    name="anime_count"
)

genre_chart = px.bar(
    genre_df.sort_values("anime_count"),
    x="anime_count",
    y="genre",
    orientation="h",
    title="Top 10 Most Common Anime Genres",
    labels={
        "anime_count": "Number of Anime",
        "genre": "Genre"
    }
)
st.plotly_chart(genre_chart, width='stretch')


# Bar graph showing similarity scores for each recommended anime
if st.session_state.current_results is not None:
    results = st.session_state.current_results

    similarity_chart = px.bar(
        results.sort_values("similarity", ascending=True),
        x="similarity",
        y="name",
        orientation="h",
        title="Similarity Scores of Recommended Anime",
        labels={
            "similarity": "Genre Similarity Score",
            "name": "Recommended Anime"
        }
    )

    similarity_chart.update_layout(
        xaxis=dict(range=[0.1, 1]),
        showlegend=False
    )

    st.plotly_chart(
        similarity_chart,
        width='stretch'
    )
