from flask import Flask, request, render_template
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import networkx as nx

app = Flask(__name__)

text = ""


@app.route("/", methods=["GET", "POST"])
def index():
    global text
    if request.method == "POST":
        text = request.form["text"]

        # Langkah 2: Tokenisasi teks
        sentences = sent_tokenize(text)

        # Langkah 4: Buat matriks kemiripan kalimat
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Langkah 5: Terapkan PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        # Langkah 6: Pilih N kalimat teratas
        ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        # Langkah 7: Cetak ringkasan
        N = 1  # jumlah kalimat dalam ringkasan
        summary = []
        for i in range(N):
            summary.append(ranked_sentences[i][1])

        return render_template("index.html", summary=" ".join(summary), text=text)
    return render_template("index.html", text=text)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run()
