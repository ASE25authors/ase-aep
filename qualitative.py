from multiprocessing import freeze_support
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from tqdm import tqdm
from wordcloud import WordCloud

def extract_llm_blocks(df_raw):
    llm_blocks = []
    for col in range(df_raw.shape[1]):
        if df_raw.iloc[1, col] == "Explanation":
            llm_col = col
            llm_name_cell = df_raw.iloc[0, col - 2] if col >= 2 else f"LLM_{col}"
            llm_name = str(llm_name_cell).strip()
            llm_explanations = df_raw.iloc[2:32, llm_col].astype(str).tolist()
            llm_blocks.append((llm_name, llm_explanations))
    df = pd.DataFrame({name: explanations for name, explanations in llm_blocks})
    return df

def extract_ethics_blocks(df_raw):
    ethics_blocks = {}
    for col in range(df_raw.shape[1]):
        if df_raw.iloc[1, col] == "Ethical Theory":
            llm_col = col
            llm_name_cell = df_raw.iloc[0, col - 3] if col >= 3 else f"LLM_{col}"
            llm_name = str(llm_name_cell).strip()
            theory = df_raw.iloc[2:32, llm_col].astype(str).tolist()
            ethics_blocks[llm_name] = theory
    return ethics_blocks

def get_explanations_per_theory(df_explanations, ethics_blocks):
    theory_explanations = {}
    for llm, theories in ethics_blocks.items():
        if llm not in df_explanations.columns:
            continue
        explanations = df_explanations[llm].dropna().astype(str).tolist()
        for t_idx, theory in enumerate(theories):
            clean_theory = str(theory).strip().lower()
            if not clean_theory or clean_theory in ['nan', 'none', '']:
                continue
            if clean_theory not in theory_explanations:
                theory_explanations[clean_theory] = []
            if t_idx < len(explanations):
                theory_explanations[clean_theory].append(explanations[t_idx])
    return theory_explanations

def generate_wordcloud(texts, out_file, title=None):
    all_text = " ".join(texts)
    wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(all_text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def main():
    file_path = "llm_outputs_parsed.xlsx"
    xls = pd.ExcelFile(file_path)
    df_raw = xls.parse('1', header=None)

    df_explanations = extract_llm_blocks(df_raw)
    explanation_columns = df_explanations.columns.tolist()
    ethics_blocks = extract_ethics_blocks(df_raw)
    print(f"Found {len(explanation_columns)} LLMs: {explanation_columns}")

    explanations = df_explanations.values.flatten()
    explanations = [str(text).strip() for text in explanations if str(text).strip()]
    if not explanations:
        raise ValueError("No valid explanations found.")

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2)
    X_tfidf = vectorizer.fit_transform(explanations)
    feature_names = vectorizer.get_feature_names_out()

    tokenized_explanations = [text.lower().split() for text in explanations]
    dictionary = Dictionary(tokenized_explanations)
    corpus = [dictionary.doc2bow(text) for text in tokenized_explanations]

    coherence_scores = []
    topic_range = range(2, 16)
    best_score, best_n = -1, 5
    print("Computing coherence scores...")
    for n in tqdm(topic_range):
        lda = LatentDirichletAllocation(n_components=n, random_state=0)
        lda.fit(X_tfidf)
        top_words = [[feature_names[i] for i in topic.argsort()[:-11:-1]] for topic in lda.components_]
        cm = CoherenceModel(topics=top_words, texts=tokenized_explanations, dictionary=dictionary, coherence='c_v', processes=1)
        score = cm.get_coherence()
        coherence_scores.append(score)
        if score > best_score:
            best_score = score
            best_n = n

    print(f"\n Best topic number: {best_n} (Coherence: {best_score:.3f})")
    for n, score in zip(topic_range, coherence_scores):
        print(f"  {n} topics: {score:.3f}")

    plt.figure(figsize=(10, 6))
    plt.plot(topic_range, coherence_scores, marker='o')
    plt.axvline(best_n, color='red', linestyle='--', label=f'Best = {best_n}')
    plt.title("LDA Coherence Score")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence (c_v)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_coherence_scores.png")
    plt.close()

    lda = LatentDirichletAllocation(n_components=best_n, random_state=0)
    lda.fit(X_tfidf)
    topic_top_words = [[feature_names[i] for i in topic.argsort()[:-11:-1]] for topic in lda.components_]
    print("\n Top 10 words per topic")
    for i, words in enumerate(topic_top_words):
        print(f"Topic {i+1}: {', '.join(words)}")

    fig, axes = plt.subplots(best_n, 1, figsize=(10, best_n * 2.2))
    for i, ax in enumerate(axes):
        words = topic_top_words[i]
        scores = lda.components_[i][[vectorizer.vocabulary_[w] for w in words]]
        ax.barh(words[::-1], scores[::-1])
        ax.set_title(f"Topic {i+1}")
        ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("plot_lda_topics.png")
    plt.close()

    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(X_tfidf)
    X_tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=0).fit_transform(X_tfidf.toarray())
    plt.figure(figsize=(12, 7))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='tab10', s=10)
    plt.title("t-SNE of Semantic Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_tsne_clusters.png")
    plt.close()

    word_counts = [len(text.split()) for text in explanations]
    sentence_counts = [text.count('.') for text in explanations]
    plt.figure(figsize=(10, 4))
    plt.hist(word_counts, bins=20, color='skyblue')
    plt.title("Word Count per Explanation")
    plt.grid(True)
    plt.savefig("plot_word_counts.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.hist(sentence_counts, bins=20, color='salmon')
    plt.title("Sentence Count per Explanation")
    plt.grid(True)
    plt.savefig("plot_sentence_counts.png")
    plt.close()

    grouped_explanations = df_explanations.apply(lambda row: [str(t).strip() for t in row if str(t).strip()], axis=1)
    similarities = []
    for group in grouped_explanations:
        if len(group) > 1:
            tfidf_group = vectorizer.transform(group)
            sim = cosine_similarity(tfidf_group)
            similarities.append(np.mean(sim[np.triu_indices_from(sim, k=1)]))
    avg_similarity = np.mean(similarities)
    print(f"\n Average similarity across answers to same question: {avg_similarity:.3f}")

    plt.figure(figsize=(6, 4))
    plt.boxplot(similarities, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    plt.title("Semantic Similarity within Each Question")
    plt.annotate(f"Mean = {avg_similarity:.2f}", xy=(1.1, avg_similarity), fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_similarity_per_question.png")
    plt.close()

    llm_similarities, llm_word_counts, llm_vectors = {}, {}, {}
    for llm in explanation_columns:
        texts = df_explanations[llm].dropna().astype(str).tolist()
        if len(texts) > 1:
            tfidf_matrix = vectorizer.transform(texts)
            llm_similarities[llm] = np.mean(cosine_similarity(tfidf_matrix)[np.triu_indices(len(texts), 1)])
            llm_word_counts[llm] = np.mean([len(t.split()) for t in texts])
            llm_vectors[llm] = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    print("\n LLM Statistics ")
    for llm in llm_similarities:
        print(f"{llm}: similarity = {llm_similarities[llm]:.3f}, avg words = {llm_word_counts[llm]:.1f}")

    sorted_llms = sorted(llm_similarities.items(), key=lambda x: x[1])
    plt.figure(figsize=(12, 6))
    plt.barh([k for k, _ in sorted_llms], [v for _, v in sorted_llms], color='mediumpurple')
    plt.title("Average Semantic Similarity per LLM")
    plt.tight_layout()
    plt.savefig("plot_llm_similarity.png")
    plt.close()

    df_stats = pd.DataFrame({
        "LLM": list(llm_similarities.keys()),
        "Similarity": list(llm_similarities.values()),
        "Words": [llm_word_counts[k] for k in llm_similarities.keys()]
    })
    plt.figure(figsize=(10, 6))
    plt.scatter(df_stats["Words"], df_stats["Similarity"], color='teal')
    for _, row in df_stats.iterrows():
        plt.text(row["Words"] + 0.3, row["Similarity"], row["LLM"], fontsize=8)
    plt.title("LLM Coherence vs Wordiness")
    plt.xlabel("Avg Word Count")
    plt.ylabel("Semantic Similarity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_llm_coherence_wordiness.png")
    plt.close()

    llm_names = list(llm_vectors.keys())
    matrix = np.vstack([llm_vectors[name] for name in llm_names])
    sim_matrix = cosine_similarity(matrix)
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, xticklabels=llm_names, yticklabels=llm_names, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Semantic Similarity Between LLMs")
    plt.tight_layout()
    plt.savefig("plot_llm_heatmap.png")
    plt.close()

    cluster = AgglomerativeClustering(n_clusters=3, metric='cosine', linkage='average')
    labels = cluster.fit_predict(matrix)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(matrix)
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("Set2", len(set(labels)))
    for i in set(labels):
        idx = np.where(labels == i)[0]
        plt.scatter(reduced[idx, 0], reduced[idx, 1], s=100, label=f"Cluster {i}", color=palette[i])
        for j in idx:
            plt.text(reduced[j, 0]+0.02, reduced[j, 1], llm_names[j], fontsize=9)
    plt.title("LLM Clustering (PCA Projection)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_llm_clustering.png")
    plt.close()

    generate_wordcloud(explanations, "wordcloud_all_llms.png", title="All LLMs - All Explanations")

    for llm in df_explanations.columns:
        texts = df_explanations[llm].dropna().astype(str).tolist()
        if len(texts) > 0:
            generate_wordcloud(
                texts,
                out_file=f"wordcloud_{llm.replace(' ', '_')}.png",
                title=f"LLM: {llm}"
            )

    theory_explanations = get_explanations_per_theory(df_explanations, ethics_blocks)
    for theory, texts in theory_explanations.items():
        if len(texts) > 0:
            safe_theory = "".join([c if c.isalnum() else "_" for c in theory])
            generate_wordcloud(
                texts,
                out_file=f"wordcloud_theory_{safe_theory}.png",
                title=f"Ethical Theory: {theory}"
            )

if __name__ == '__main__':
    freeze_support()
    main()