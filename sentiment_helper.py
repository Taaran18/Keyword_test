from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def assign_cluster_sentiment(
    df, positive_definition: str, negative_definition: str, threshold=0.05
):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pos_vec = model.encode([positive_definition])[0].reshape(1, -1)
    neg_vec = model.encode([negative_definition])[0].reshape(1, -1)

    sentiments = []
    for desc in df["Intent_Description"]:
        if desc == "N/A":
            sentiments.append("Negative")
            continue
        desc_vec = model.encode([desc])[0].reshape(1, -1)
        pos_sim = cosine_similarity(desc_vec, pos_vec)[0][0]
        neg_sim = cosine_similarity(desc_vec, neg_vec)[0][0]

        if abs(pos_sim - neg_sim) < threshold:
            sentiments.append("Positive")  # Treat borderline neutral as Positive
        else:
            sentiments.append("Positive" if pos_sim > neg_sim else "Negative")

    df["Cluster_Sentiment"] = sentiments
    return df
