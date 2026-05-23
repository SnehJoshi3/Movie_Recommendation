# Movie Matcher AI 🎬

A full-stack content-based movie recommendation system built using **Flask**, **Machine Learning**, and **TMDB movie datasets**.
The system recommends similar movies based on textual metadata using **TF-IDF Vectorization** and **Cosine Similarity** while also providing interactive analytics dashboards and dataset exploration tools.

---

## 🚀 Project Overview

Movie Matcher AI is an academic-level recommendation engine that allows users to search for movies and receive similar recommendations instantly.

The project focuses on:

* Fast recommendation generation
* Clean UI/UX
* Explainable ML workflow
* Dataset analytics & visualization
* Scalable preprocessing pipeline

Unlike basic static recommenders, this project includes:

* Backend optimization
* Dashboard analytics
* Recommendation corpus analysis
* Evolution & architecture explanation pages
* Interactive frontend integration

---

## 🧠 Recommendation System Type

This project uses:

### Content-Based Filtering

Recommendations are generated using:

* Genres
* Movie overviews
* Keywords
* Cast
* Director metadata

The system compares movie similarity using:

* **TF-IDF Vectorization**
* **Cosine Similarity**

---

## ⚙️ Tech Stack

### Backend

* Python
* Flask
* SQLite3
* Pandas
* Scikit-learn
* Pickle

### Frontend

* HTML5
* CSS3
* JavaScript
* Jinja2 Templates
* Chart.js

### ML Concepts

* TF-IDF Vectorization
* Cosine Similarity
* Sparse Matrices
* Text Preprocessing
* Feature Engineering

---

## 📂 Dataset Used

### Primary Dataset

* TMDB Full Movies Dataset (1M+ Movies) ([Kaggle][1])

Dataset Link:
[Kaggle Dataset - TMDB Movies Dataset 2024](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies?utm_source=chatgpt.com)

### Processed Recommendation Corpus

* Curated Top 10,000 movies
* Selected using vote count & metadata quality
* Optimized for fast inference and memory efficiency

---

## 🔍 How Recommendation Works

### Workflow

1. Raw TMDB dataset loaded
2. Data cleaning & preprocessing
3. Feature engineering performed
4. Metadata combined into tags
5. TF-IDF vectorization applied
6. Sparse matrix generated
7. TF-IDF vectors serialized using Pickle
8. Flask server loads precomputed vectors
9. Cosine similarity calculated dynamically
10. Top similar movies returned to user

---

## 📊 Features

### 🎥 Movie Recommendations

* Search-based recommendations
* Similar movie generation
* Poster & metadata display

### 📈 Dashboard Analytics

* Raw dataset insights
* Genre distribution
* Language analysis
* Popularity metrics
* Director statistics

### 🧪 Analysis Module

* Exploratory Data Analysis (EDA)
* Processed recommendation dataset analysis
* Recommendation corpus visualization

### 🧬 Evolution Page

Project development journey from:

* Google Colab notebook
  → Flask application
  → Optimized ML system
  → Full analytics platform

### ⚡ Optimization Features

* Pickle serialization
* Precomputed vectors
* In-memory inference
* Faster startup performance

---

## 🧮 Machine Learning Concepts Used

### TF-IDF (Term Frequency–Inverse Document Frequency)

Converts textual movie metadata into weighted numerical vectors.

### Cosine Similarity

Measures similarity between movie vectors using angular similarity.

### Sparse Matrix

Efficiently stores TF-IDF vectors by saving only non-zero values.

---

## 🏗️ Project Architecture

```plaintext
User Input
    ↓
Flask Backend
    ↓
Movie Lookup
    ↓
TF-IDF Vector Retrieval
    ↓
Cosine Similarity Calculation
    ↓
Top Similar Movies
    ↓
Frontend Rendering
```

---

## 📌 Future Improvements

* Hybrid Recommendation Systems
* User Authentication
* Watch History Tracking
* Personalized Recommendations
* Trailer Integration
* Streaming Platform Integration
* Vector Database Integration
* Approximate Nearest Neighbor Search (FAISS)

---

## ⚠️ Current Limitations

* Recommendations limited to processed corpus
* No collaborative filtering
* Limited personalization
* Hollywood-heavy dataset bias
* Static preprocessing pipeline

---

## 🖥️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/movie-matcher-ai.git
cd movie-matcher-ai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Flask Server

```bash
python app.py
```

### Open Browser

```plaintext
http://127.0.0.1:5000
```

---

## 📚 Learning Outcomes

This project helped in understanding:

* End-to-end ML workflows
* Recommendation systems
* Flask integration
* Data preprocessing
* Dashboard analytics
* Performance optimization
* Full-stack ML deployment concepts

---

## 🙌 Acknowledgements

* TMDB Dataset Contributors
* Kaggle Dataset Community
* Scikit-learn Documentation
* Flask Documentation

---

## 📄 License

This project is intended for:

* Academic purposes
* Learning
* Research
* Portfolio demonstration

TMDB dataset rights belong to their respective owners.

[1]: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies?utm_source=chatgpt.com "Full TMDB Movies Dataset 2024 (1M Movies)"
