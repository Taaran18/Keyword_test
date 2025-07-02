import re
import os
import warnings
import pandas as pd
from sklearn.cluster import KMeans
from models import load_question_classifier, load_embedding_model

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def load_and_clean_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print("✅ File loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None
    columns_to_drop = [
        "Three month change", "YoY change", "Competition",
        "Competition (indexed value)", "Ad impression share",
        "Organic impression share", "Organic average position", "In account?",
        "In plan?", "Searches: May 2024", "Searches: Jun 2024", "Searches: Jul 2024",
        "Searches: Aug 2024", "Searches: Sep 2024", "Searches: Oct 2024",
        "Searches: Nov 2024", "Searches: Dec 2024", "Searches: Jan 2025",
        "Searches: Feb 2025", "Searches: Mar 2025", "Searches: Apr 2025", "Searches: May 2025",
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns],
            inplace=True, errors="ignore")
    return df

def filter_questions(df: pd.DataFrame) -> tuple:
    classifier = load_question_classifier()
    df["is_question"] = df["Keyword"].apply(
        lambda x: max(classifier(x)[0], key=lambda y: y["score"])["label"]
    )
    df["is_question"] = df["is_question"].apply(
        lambda x: "Ques" if x == "LABEL_1" else "Not Ques"
    )
    df_removed_classifier = df[df["is_question"] == "Ques"].copy()
    df_remaining = df[df["is_question"] != "Ques"].copy()

    # Extended set of question starters
    question_starters = {
        "can", "could", "do", "does", "did", "what", "how", "where", "why", "who",
        "will", "shall", "is", "are", "was", "were", "would", "should", "if",
        "may", "might", "must", "tell me", "explain", "describe"
    }
    pattern = r"^(" + "|".join(re.escape(word) for word in question_starters) + r")\b"
    df_removed_manual = df_remaining[
        df_remaining["Keyword"].str.strip().str.lower().str.match(pattern)
    ].copy()
    df_removed_manual["is_question"] = "Ques"
    df_final_filtered = df_remaining.drop(df_removed_manual.index)

    df_removed_total = pd.concat(
        [df_removed_classifier, df_removed_manual], ignore_index=True
    )
    return df_final_filtered, df_removed_total


def filter_patterns(df: pd.DataFrame, words_to_filter: list) -> tuple:
    if not words_to_filter:
        return df, pd.DataFrame()
    pattern = "|".join([w.strip() for w in words_to_filter if w.strip()])
    df_removed = df[df["Keyword"].str.contains(pattern, case=False, na=False)]
    df_filtered = df.drop(df_removed.index)
    return df_filtered, df_removed

def cluster_keywords(df, min_k, max_k):
    model = load_embedding_model()
    keywords = df["Keyword"].tolist()
    embeddings = model.encode(keywords, show_progress_bar=True)
    total_keywords = len(keywords)
    approx_groups = max(1, total_keywords // max_k)
    kmeans = KMeans(n_clusters=approx_groups * 2, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    df["TempCluster"] = labels

    final_groups = []
    misc_group = []

    for _, group_df in df.groupby("TempCluster"):
        keyword_list = group_df["Keyword"].tolist()
        if len(keyword_list) < min_k:
            misc_group.extend(keyword_list)
        elif len(keyword_list) > max_k:
            for i in range(0, len(keyword_list), max_k):
                chunk = keyword_list[i: i + max_k]
                if len(chunk) < min_k:
                    misc_group.extend(chunk)
                else:
                    final_groups.append(chunk)
        else:
            final_groups.append(keyword_list)

    clustered_data = []
    for i, group in enumerate(final_groups):
        for kw in group:
            clustered_data.append({"Keyword": kw, "Cluster": i})

    df_clustered = pd.DataFrame(clustered_data)
    df_misc = pd.DataFrame({"Keyword": misc_group})
    return df_clustered, embeddings, kmeans.cluster_centers_, df_misc
