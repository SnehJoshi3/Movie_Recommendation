## Movie Matcher AI
An end-to-end, full-stack content-based movie recommendation engine and analytics platform. Built to process over 1.16 million raw TMDB records down to an optimized 10,000-movie recommendation corpus utilizing TF-IDF vectorization, Cosine Similarity, and interactive Chart.js dashboards.

---

## Tech Stack & Tools

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)

---

##  Detailed Project Description

**Movie Matcher AI** bridges the gap between raw data science research and an interactive, production-ready web application. Traditional recommenders often operate as static Jupyter Notebooks; this platform features an **optimized backend pipeline** paired with an **executive analytics dashboard**.

By processing textual metadata—including overviews, genres, cast, crew, and keywords—the platform computes dynamic high-dimensional vector spaces using **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity**. The serialized sparse matrix allows the Flask server to deliver real-time recommendations without re-computing vectors on every incoming request.

---

## 📸 Modules & Dashboard Previews

### 1. Recommendation Engine Module
Users can search for any movie in the curated corpus to receive instant content-based recommendations, complete with poster metadata, genre tags, and similarity context.

<img width="1900" height="863" alt="image" src="https://github.com/user-attachments/assets/75a4212b-d5f7-4a90-990b-17bb52ea0ccb" />
<img width="1155" height="810" alt="image" src="https://github.com/user-attachments/assets/09993b63-1005-49ac-9d54-1625766ab07d" />

### 2.  Dataset Analytics Dashboard Module
An interactive dashboard displaying global dataset distribution metrics, top genres across 19 categories, language diversity across 178 spoken languages, and vote-weight histograms.

<img width="1897" height="858" alt="image" src="https://github.com/user-attachments/assets/1454740e-4a91-4df7-9b62-69f49056c743" />

### 3. Exploratory Data Analysis (EDA) Module
Visual breakdown of the data cleaning pipeline: analyzing missing value distributions, filtering mechanisms, and the reduction of 1.16M uncurated records down to the top 10K recommendation corpus.

<img width="1903" height="860" alt="image" src="https://github.com/user-attachments/assets/f7cadd7d-59fd-439f-8c1c-40a1caf2809e" />

### 4.  Application Architecture & Evolution Page
Documents the full lifecycle of the platform—tracing its growth from a prototype Google Colab notebook to a modular Flask application backed by SQLite3 and precomputed vector stores.

<img width="1901" height="868" alt="image" src="https://github.com/user-attachments/assets/0f7ffc62-366c-488b-bdc8-056680dd38a9" />

## Dataset Link:
https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
