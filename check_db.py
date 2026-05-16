import sqlite3
conn = sqlite3.connect('movies_database.db')
rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", [r[0] for r in rows])
for tbl in [r[0] for r in rows]:
    cnt = conn.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl}: {cnt} rows")
conn.close()
