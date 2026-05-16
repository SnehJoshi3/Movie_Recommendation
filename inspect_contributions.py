import sqlite3

conn = sqlite3.connect('contributions.db')
cur = conn.cursor()

print("=== contributions.db schema ===")
cur.execute("PRAGMA table_info(contributions)")
for col in cur.fetchall():
    print(f"  col {col[0]}: {col[1]} ({col[2]})")

print("\n=== All rows ===")
cur.execute("SELECT id, movie_name, tmdb_imdb_link, added_at FROM contributions ORDER BY added_at DESC")
rows = cur.fetchall()
if rows:
    for r in rows:
        print(f"  [{r[0]}] {r[1]} | {r[2]} | {r[3]}")
else:
    print("  (no rows yet)")

print(f"\nTotal: {len(rows)} contribution(s)")
conn.close()
