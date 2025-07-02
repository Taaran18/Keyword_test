import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def plot_intent_distribution(df):
    intent_counts = df["Intent_Type"].value_counts()
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(
        intent_counts.index, intent_counts.values, color="#4a90e2", edgecolor="black"
    )
    ax.set_title(
        "Distribution of Search Intent Types", fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xlabel("Intent Type", fontsize=12)
    ax.set_ylabel("Number of Keywords", fontsize=12)
    ax.set_xticks(range(len(intent_counts)))
    ax.set_xticklabels(intent_counts.index, rotation=45, ha="right")
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            str(height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    st.pyplot(fig)


def export_data_to_excel(dfs_dict: dict, file_name="keyword_intelligence.xlsx"):
    buffer = io.BytesIO()
    max_k = st.session_state.get("max_keywords_per_group", 100)
    used_sheet_names = set()

    def get_unique_sheet_name(base_name):
        name = (
            base_name.strip()
            .replace("/", "-")
            .replace("\\", "-")
            .replace(" ", "_")[:25]
        )
        safe_name = name
        i = 1
        while safe_name.lower() in used_sheet_names:
            safe_name = f"{name}_{i}"
            i += 1
        used_sheet_names.add(safe_name.lower())
        return safe_name[:31]

    def drop_cluster_column(df):
        return df.drop(columns=["Cluster"], errors="ignore")

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for sheet_name, df in dfs_dict.items():
            if df is not None and not df.empty:
                safe_name = get_unique_sheet_name(sheet_name)
                df_to_export = drop_cluster_column(df.copy())
                df_to_export.to_excel(writer, sheet_name=safe_name, index=False)

        final_df = dfs_dict.get("Final_Clustered_Keywords")
        if final_df is not None and "Intent_Type" in final_df.columns:
            for intent_type, group_df in final_df.groupby("Intent_Type"):
                chunks = [
                    group_df[i : i + max_k] for i in range(0, len(group_df), max_k)
                ]
                for idx, chunk in enumerate(chunks):
                    base = intent_type if idx == 0 else f"{intent_type}_{idx}"
                    safe_sheet = get_unique_sheet_name(base)
                    chunk = drop_cluster_column(chunk.copy())
                    chunk.to_excel(writer, sheet_name=safe_sheet, index=False)

        if final_df is not None and "Cluster_Sentiment" in final_df.columns:
            pos_clusters = final_df[final_df["Cluster_Sentiment"] == "Positive"]
            neg_clusters = final_df[final_df["Cluster_Sentiment"] == "Negative"]
            if not pos_clusters.empty:
                pos_clusters = drop_cluster_column(pos_clusters.copy())
                pos_clusters.to_excel(
                    writer,
                    sheet_name=get_unique_sheet_name("Cluster_Positive"),
                    index=False,
                )
            if not neg_clusters.empty:
                neg_clusters = drop_cluster_column(neg_clusters.copy())
                neg_clusters.to_excel(
                    writer,
                    sheet_name=get_unique_sheet_name("Cluster_Negative"),
                    index=False,
                )

        misc_df = st.session_state.get("misc_keywords")
        if misc_df is not None and not misc_df.empty:
            misc_df = drop_cluster_column(misc_df.copy())
            misc_df.to_excel(
                writer,
                sheet_name=get_unique_sheet_name("Miscellaneous_Keywords"),
                index=False,
            )

    buffer.seek(0)
    return buffer
