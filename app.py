from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

app = Flask(__name__)

# ── Load pickled model data once at startup ──────────────────────────────────
movies_list = pickle.load(open(get_path('movies_dict.pkl'), 'rb'))
vectorized_data = pickle.load(open(get_path('vectorized_data.pkl'), 'rb'))
movies = pd.DataFrame(movies_list)
titles = movies['title'].tolist()

# ── CSV columns needed for dashboard (avoid loading all 28 cols) ─────────────
_CSV_COLS = ['title', 'vote_average', 'vote_count', 'release_date',
             'original_language', 'genres', 'cast', 'director', 'popularity']
_LANG_MAP  = {
    'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French',
    'ja': 'Japanese', 'ko': 'Korean', 'de': 'German', 'it': 'Italian',
    'zh': 'Chinese', 'pt': 'Portuguese'
}

# ── Pre-compute all dashboard stats ONCE at startup (from full raw CSV) ───────
def _build_dashboard_stats():
    csv_path   = get_path('TMDB_all_movies.csv')
    CHUNK      = 100_000          # rows per chunk — keeps RAM low
    RATING_BINS   = [0, 2, 4, 5, 6, 7, 8, 9, 10]
    RATING_LABELS = ['0-2', '2-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10']
    VOTE_BINS     = [0, 50, 200, 500, 1000, 5000, float('inf')]
    VOTE_LABELS   = ['<50', '50-200', '200-500', '500-1K', '1K-5K', '5K+']

    # Accumulators
    total_rows   = 0
    genre_ctr    : dict = {}
    lang_ctr     : dict = {}
    rating_ctr   = {l: 0 for l in RATING_LABELS}
    decade_ctr   : dict = {}
    cast_ctr     : dict = {}
    dir_ctr      : dict = {}
    vote_ctr     = {l: 0 for l in VOTE_LABELS}
    null_ctr     : dict = {}
    rating_sum   = 0.0
    rating_n     = 0
    pop_sum      = 0.0
    pop_n        = 0
    pop_all      = []          # for quartile computation (sampled)
    top_movie_title  = 'N/A'
    top_movie_rating = -1.0
    unique_langs_set : set = set()
    unique_dirs_set  : set = set()

    # columns that exist in CSV (determined on first chunk)
    _cols_checked = False
    _has = {}

    for chunk in pd.read_csv(csv_path, usecols=_CSV_COLS,
                              chunksize=CHUNK, low_memory=False,
                              encoding='utf-8', on_bad_lines='skip'):
        total_rows += len(chunk)

        # Determine which columns are available (once)
        if not _cols_checked:
            _has = {c: c in chunk.columns for c in _CSV_COLS}
            for c in _CSV_COLS:
                null_ctr[c] = 0
            _cols_checked = True

        # ── Null counts ───────────────────────────────────────────────────────
        for c in _CSV_COLS:
            if _has.get(c):
                null_ctr[c] = null_ctr.get(c, 0) + int(chunk[c].isnull().sum())

        # ── Genres ────────────────────────────────────────────────────────────
        if _has.get('genres'):
            for g in chunk['genres'].dropna():
                for part in str(g).split(','):
                    part = part.strip()
                    if part:
                        genre_ctr[part] = genre_ctr.get(part, 0) + 1

        # ── Language ──────────────────────────────────────────────────────────
        if _has.get('original_language'):
            for code, cnt in chunk['original_language'].dropna().value_counts().items():
                lang_ctr[code] = lang_ctr.get(code, 0) + int(cnt)
                unique_langs_set.add(code)

        # ── Rating distribution ───────────────────────────────────────────────
        if _has.get('vote_average'):
            va = pd.to_numeric(chunk['vote_average'], errors='coerce').dropna()
            rated = pd.cut(va, bins=RATING_BINS, labels=RATING_LABELS)
            for lbl, cnt in rated.value_counts().items():
                rating_ctr[lbl] = rating_ctr.get(lbl, 0) + int(cnt)
            rating_sum += float(va.sum())
            rating_n   += len(va)
            # top-rated movie
            local_max_idx = va.idxmax() if len(va) else None
            if local_max_idx is not None:
                local_max = va[local_max_idx]
                if local_max > top_movie_rating and _has.get('title'):
                    top_movie_rating = float(local_max)
                    top_movie_title  = str(chunk.loc[local_max_idx, 'title'])

        # ── Decade ────────────────────────────────────────────────────────────
        if _has.get('release_date'):
            years = pd.to_datetime(chunk['release_date'], errors='coerce').dt.year.dropna()
            for yr in years:
                dec = f"{int(yr) // 10 * 10}s"
                decade_ctr[dec] = decade_ctr.get(dec, 0) + 1

        # ── Cast ──────────────────────────────────────────────────────────────
        if _has.get('cast'):
            for c in chunk['cast'].dropna():
                for person in str(c).split(','):
                    person = person.strip()
                    if person:
                        cast_ctr[person] = cast_ctr.get(person, 0) + 1

        # ── Directors ─────────────────────────────────────────────────────────
        if _has.get('director'):
            for d in chunk['director'].dropna():
                for name in str(d).split(','):
                    name = name.strip()
                    if name:
                        dir_ctr[name]      = dir_ctr.get(name, 0) + 1
                        unique_dirs_set.add(name)

        # ── Popularity ────────────────────────────────────────────────────────
        if _has.get('popularity'):
            pp = pd.to_numeric(chunk['popularity'], errors='coerce').dropna()
            pop_sum += float(pp.sum())
            pop_n   += len(pp)
            # Keep a random 1-in-10 sample for quartile estimation
            pop_all.extend(pp.iloc[::10].tolist())

        # ── Vote count buckets ────────────────────────────────────────────────
        if _has.get('vote_count'):
            vc = pd.to_numeric(chunk['vote_count'], errors='coerce').dropna()
            bucketed = pd.cut(vc, bins=VOTE_BINS, labels=VOTE_LABELS)
            for lbl, cnt in bucketed.value_counts().items():
                vote_ctr[lbl] = vote_ctr.get(lbl, 0) + int(cnt)

    # ── Post-loop aggregation ─────────────────────────────────────────────────

    # Genre
    genre_series  = pd.Series(genre_ctr).sort_values(ascending=False)
    genre_counts  = genre_series.head(12)
    unique_genres = int(len(genre_series))

    # Language
    lang_series = pd.Series(lang_ctr).sort_values(ascending=False)
    lang_top    = lang_series.head(6)
    lang_labels = [_LANG_MAP.get(c, c.upper()) for c in lang_top.index]
    lang_values = [int(v) for v in lang_top.values]

    # Rating
    rating_labels = RATING_LABELS
    rating_values = [rating_ctr[l] for l in RATING_LABELS]
    avg_rating    = round(rating_sum / rating_n, 2) if rating_n else 0

    # Decade
    decade_labels = sorted(decade_ctr.keys())
    decade_values = [decade_ctr[d] for d in decade_labels]

    # Cast top-10
    cast_series      = pd.Series(cast_ctr).sort_values(ascending=False)
    top_cast         = cast_series.head(10)
    top_cast_labels  = list(top_cast.index)
    top_cast_values  = [int(v) for v in top_cast.values]

    # Directors top-10
    dir_series    = pd.Series(dir_ctr).sort_values(ascending=False)
    top_dir       = dir_series.head(10)
    top_directors = [{'name': n, 'count': int(c)} for n, c in top_dir.items()]

    # Popularity
    avg_pop = round(pop_sum / pop_n, 2) if pop_n else 0
    pop_labels, pop_values = [], []
    if pop_all:
        pop_s = pd.Series(pop_all)
        q1, q2, q3 = pop_s.quantile([0.25, 0.5, 0.75])
        pop_labels = ['Low (<Q1)', 'Medium (Q1-Q2)', 'High (Q2-Q3)', 'Viral (>Q3)']
        pop_values = [
            int((pop_s < q1).sum()),
            int(((pop_s >= q1) & (pop_s < q2)).sum()),
            int(((pop_s >= q2) & (pop_s < q3)).sum()),
            int((pop_s >= q3).sum())
        ]

    # Vote count
    vote_labels = VOTE_LABELS
    vote_values = [vote_ctr[l] for l in VOTE_LABELS]

    # Null analysis
    null_series = pd.Series(null_ctr).sort_values(ascending=False)
    null_nonzero = null_series[null_series > 0].head(8)
    if null_nonzero.empty:
        cols_to_show = [c for c in ['title', 'overview', 'genres', 'cast', 'director',
                                     'vote_average', 'vote_count', 'popularity',
                                     'original_language'] if c in null_ctr]
        null_labels = cols_to_show
        null_values = [round((1 - null_ctr[c] / max(total_rows, 1)) * 100, 1) for c in cols_to_show]
        _null_mode  = 'completeness'
    else:
        null_labels = list(null_nonzero.index)
        null_values = [int(v) for v in null_nonzero.values]
        _null_mode  = 'missing'

    return {
        'stats': {
            'total_csv':     f"{total_rows:,}",
            'unique_genres': f"{unique_genres:,}",
            'unique_langs':  f"{len(unique_langs_set):,}",
            'unique_dirs':   f"{len(unique_dirs_set):,}",
            'avg_rating':    f"{avg_rating}",
            'top_movie':     top_movie_title,
            'avg_pop':       f"{avg_pop:,.1f}",
        },
        'charts': {
            'genre_json':  json.dumps({'labels': list(genre_counts.index),
                                        'values': [int(v) for v in genre_counts.values]}),
            'lang_json':   json.dumps({'labels': lang_labels, 'values': lang_values}),
            'rating_json': json.dumps({'labels': rating_labels, 'values': rating_values}),
            'decade_json': json.dumps({'labels': decade_labels, 'values': decade_values}),
            'cast_json':   json.dumps({'labels': top_cast_labels, 'values': top_cast_values}),
            'pop_json':    json.dumps({'labels': pop_labels, 'values': pop_values}),
            'vote_json':   json.dumps({'labels': vote_labels, 'values': vote_values}),
            'null_json':   json.dumps({'labels': null_labels, 'values': null_values,
                                        'mode': _null_mode}),
        },
        'top_directors': top_directors,
    }

# Build once at startup — reads full raw CSV in chunks (memory-efficient)
print('[Dashboard] Computing stats from full raw CSV … this may take ~30 s on first start.')
_DASHBOARD_DATA = _build_dashboard_stats()
print('[Dashboard] Stats ready.')

# ─────────────────────────────────────────────────────────────────────────────

def get_filtered_data(filter_type, value, limit=10):
    if filter_type == 'genre':
        filtered = movies[movies['genres'].str.contains(value, case=False, na=False)]
    elif filter_type == 'language':
        filtered = movies[movies['original_language'] == value]
    elif filter_type == 'popular':
        filtered = movies.sort_values(by='popularity', ascending=False)
    else:
        filtered = movies

    top_list = filtered.sort_values(by="vote_average", ascending=False).head(limit)

    results = []
    for i in range(len(top_list)):
        results.append({
            'title':  top_list.iloc[i].title,
            'poster': f"https://image.tmdb.org/t/p/w500{top_list.iloc[i].poster_path}",
            'rating': top_list.iloc[i].vote_average
        })
    return results


@app.route('/')
def index():
    data = {
        'action':   get_filtered_data('genre', 'Action'),
        'comedy':   get_filtered_data('genre', 'Comedy'),
        'musical':  get_filtered_data('genre', 'Music'),
        'hindi':    get_filtered_data('language', 'hi'),
        'english':  get_filtered_data('language', 'en'),
        'thrillers': get_filtered_data('genre', 'Thriller')
    }
    return render_template('index.html', titles=titles, **data)


@app.route('/analysis')
def analysis():
    df = movies.copy()

    # ── Genre distribution (doughnut) ──────────────────────────────────────────
    all_genres = []
    for g in df['genres'].dropna():
        all_genres.extend([x.strip() for x in str(g).split(',') if x.strip()])
    genre_series = pd.Series(all_genres).value_counts().head(8)
    genre_json = json.dumps({'labels': list(genre_series.index),
                             'values': [int(v) for v in genre_series.values]})

    # ── Rating distribution (bar) ──────────────────────────────────────────────
    r_bins   = [0, 2, 4, 5, 6, 7, 8, 9, 10]
    r_labels = ['0-2', '2-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10']
    r_counts = pd.cut(pd.to_numeric(df['vote_average'], errors='coerce').dropna(),
                      bins=r_bins, labels=r_labels).value_counts().sort_index()
    rating_json = json.dumps({'labels': list(r_counts.index.astype(str)),
                              'values': [int(v) for v in r_counts.values]})

    # ── Language breakdown (pie) ───────────────────────────────────────────────
    lang_map = {
        'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French',
        'ja': 'Japanese', 'ko': 'Korean', 'de': 'German', 'it': 'Italian',
        'zh': 'Chinese', 'pt': 'Portuguese'
    }
    lang_top = df['original_language'].value_counts().head(8)
    lang_json = json.dumps({
        'labels': [lang_map.get(c, c.upper()) for c in lang_top.index],
        'values': [int(v) for v in lang_top.values]
    })

    # ── Popularity tiers (polar) ───────────────────────────────────────────────
    pop = pd.to_numeric(df['popularity'], errors='coerce').dropna()
    q1, q2, q3 = pop.quantile([0.25, 0.5, 0.75])
    pop_json = json.dumps({
        'labels': ['Low (<Q1)', 'Medium (Q1-Q2)', 'High (Q2-Q3)', 'Viral (>Q3)'],
        'values': [
            int((pop < q1).sum()),
            int(((pop >= q1) & (pop < q2)).sum()),
            int(((pop >= q2) & (pop < q3)).sum()),
            int((pop >= q3).sum())
        ]
    })

    # ── Vote count buckets (horizontal bar) ───────────────────────────────────
    vc = pd.to_numeric(df['vote_count'], errors='coerce').dropna()
    vc_bins   = [0, 50, 200, 500, 1000, 5000, float('inf')]
    vc_labels = ['<50', '50-200', '200-500', '500-1K', '1K-5K', '5K+']
    vc_counts = pd.cut(vc, bins=vc_bins, labels=vc_labels).value_counts().sort_index()
    vote_json = json.dumps({'labels': list(vc_counts.index.astype(str)),
                            'values': [int(v) for v in vc_counts.values]})

    # ── Top 10 directors by movie count (horizontal bar) ──────────────────────
    dir_counts = df['director'].dropna().value_counts().head(10)
    director_json = json.dumps({'labels': list(dir_counts.index),
                                'values': [int(v) for v in dir_counts.values]})

    # ── Cast depth: how many cast members per movie (histogram) ───────────────
    cast_depths = df['cast'].dropna().apply(
        lambda x: len([p for p in str(x).split(',') if p.strip()]))
    cd_bins   = [0, 1, 2, 3, 5, 8, 12, float('inf')]
    cd_labels = ['0', '1', '2', '3-4', '5-7', '8-11', '12+']
    cd_counts = pd.cut(cast_depths, bins=cd_bins, labels=cd_labels).value_counts().sort_index()
    cast_depth_json = json.dumps({'labels': list(cd_counts.index.astype(str)),
                                  'values': [int(v) for v in cd_counts.values]})

    # ── Data completeness (horizontal bar) ────────────────────────────────────
    key_cols = ['title', 'genres', 'cast', 'director', 'vote_average',
                'vote_count', 'popularity', 'original_language', 'overview']
    key_cols = [c for c in key_cols if c in df.columns]
    completeness = (df[key_cols].notnull().mean() * 100).round(1)
    completeness_json = json.dumps({'labels': list(completeness.index),
                                    'values': [float(v) for v in completeness.values]})

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    kpis = {
        'total':       f"{len(df):,}",
        'avg_rating':  f"{pd.to_numeric(df['vote_average'], errors='coerce').mean():.2f}",
        'unique_lang': f"{df['original_language'].nunique()}",
        'unique_dir':  f"{df['director'].nunique():,}",
        'unique_genre':f"{pd.Series(all_genres).nunique()}",
        'avg_pop':     f"{pop.mean():.1f}",
    }

    return render_template('analysis.html',
                           genre_json=genre_json,
                           rating_json=rating_json,
                           lang_json=lang_json,
                           pop_json=pop_json,
                           vote_json=vote_json,
                           director_json=director_json,
                           cast_depth_json=cast_depth_json,
                           completeness_json=completeness_json,
                           kpis=kpis)


@app.route('/recommend', methods=['POST'])
def recommend():
    selected_movie = request.form.get('movie_name')

    data = {
        'action':   get_filtered_data('genre', 'Action'),
        'comedy':   get_filtered_data('genre', 'Comedy'),
        'musical':  get_filtered_data('genre', 'Music'),
        'hindi':    get_filtered_data('language', 'hi'),
        'english':  get_filtered_data('language', 'en'),
        'thrillers': get_filtered_data('genre', 'Thriller')
    }

    if selected_movie not in movies['title'].values:
        return render_template('index.html', error="Movie not found!", titles=titles, **data)

    idx = movies[movies['title'] == selected_movie].index[0]
    similarity_scores = cosine_similarity(vectorized_data[idx], vectorized_data).flatten()
    similar_indices = similarity_scores.argsort()[-11:][::-1]   # +1 to skip self

    recommendations = []
    for i in similar_indices:
        if movies.iloc[i].title == selected_movie:
            continue
        recommendations.append({
            'title':  movies.iloc[i].title,
            'poster': f"https://image.tmdb.org/t/p/w500{movies.iloc[i].poster_path}",
            'rating': movies.iloc[i].vote_average,
        })
        if len(recommendations) == 10:
            break

    return render_template('index.html', titles=titles,
                           results=recommendations, original=selected_movie, **data)


@app.route('/dashboard')
def dashboard():
    """Serves pre-computed stats — instant response, no CSV I/O."""
    conn = sqlite3.connect(get_path('contributions.db'))
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT count(*) FROM contributions")
        contributed_count = cursor.fetchone()[0]
    except Exception:
        contributed_count = 0
    conn.close()

    data = dict(_DASHBOARD_DATA)          # shallow copy
    data['stats'] = dict(data['stats'])   # don't mutate cached dict
    data['stats']['contributed'] = f"{contributed_count:,}"
    return render_template('dashboard.html', **data)


@app.route('/contribute', methods=['GET', 'POST'])
def contribute():
    """CRUD: Reads and Inserts user-submitted movies into the contributions table."""
    conn = sqlite3.connect(get_path('contributions.db'))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contributions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_name     TEXT    NOT NULL,
            tmdb_imdb_link TEXT    NOT NULL,
            added_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    message = None
    error   = None

    if request.method == 'POST':
        movie_name     = request.form.get('movie_name', '').strip()
        tmdb_imdb_link = request.form.get('tmdb_imdb_link', '').strip()

        if not movie_name or not tmdb_imdb_link:
            error = "Movie Name and TMDB/IMDB link are required."
        else:
            try:
                cursor.execute(
                    "INSERT INTO contributions (movie_name, tmdb_imdb_link) VALUES (?, ?)",
                    (movie_name, tmdb_imdb_link)
                )
                conn.commit()
                message = f"'{movie_name}' has been successfully submitted. Thank you!"
            except Exception as ex:
                error = f"Database error: {ex}"

    # Fetch only the 5 most recent contributions
    cursor.execute(
        "SELECT id, movie_name, tmdb_imdb_link, added_at FROM contributions ORDER BY added_at DESC LIMIT 5"
    )
    contributions = cursor.fetchall()
    conn.close()

    return render_template('contribute.html', message=message, error=error, contributions=contributions)


@app.route('/concepts')
def concepts():
    """Renders the How It Works / ML pipeline education page."""
    return render_template('concepts.html')


@app.route('/evolution')
def evolution():
    """Renders the Project Evolution / Development Journey page."""
    return render_template('evolution.html')


if __name__ == '__main__':
    app.run(debug=True)