## Movie Matcher AI
An end-to-end, full-stack content-based movie recommendation engine and analytics platform. Built to process over 1.16 million raw TMDB records down to an optimized 10,000-movie recommendation corpus utilizing TF-IDF vectorization, Cosine Similarity, and interactive Chart.js dashboards.

---

## Tech Stack & Tools

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)]()
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)

---

##  Detailed Project Description

**Movie Matcher AI** bridges the gap between raw data science research and an interactive, production-ready web application. Traditional recommenders often operate as static Jupyter Notebooks; this platform features an **optimized backend pipeline** paired with an **executive analytics dashboard**.

By processing textual metadata—including overviews, genres, cast, crew, and keywords—the platform computes dynamic high-dimensional vector spaces using **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity**. The serialized sparse matrix allows the Flask server to deliver real-time recommendations without re-computing vectors on every incoming request.

---

## 📸 Modules & Dashboard Previews

### 1. 🎥 Recommendation Engine Module
Users can search for any movie in the curated corpus to receive instant content-based recommendations, complete with poster metadata, genre tags, and similarity context.

![Recommendation Engine Preview](assets/recommendation-module.png)

---

### 2. 📈 Dataset Analytics Dashboard Module
An interactive dashboard displaying global dataset distribution metrics, top genres across 19 categories, language diversity across 178 spoken languages, and vote-weight histograms.

![Analytics Dashboard Preview](assets/dashboard-module.png)

---

### 3. 🧪 Exploratory Data Analysis (EDA) Module
Visual breakdown of the data cleaning pipeline: analyzing missing value distributions, filtering mechanisms, and the reduction of 1.16M uncurated records down to the top 10K recommendation corpus.

![EDA Module Preview](assets/eda-module.png)

---

### 4. 🧬 Application Architecture & Evolution Page
Documents the full lifecycle of the platform—tracing its growth from a prototype Google Colab notebook to a modular Flask application backed by SQLite3 and precomputed vector stores.

![Evolution Page Preview](assets/evolution-module.png)

---
