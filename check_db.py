import sqlite3

conn = sqlite3.connect("data/face_db.db")
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = c.fetchall()
print("All tables:", [t[0] for t in tables])

for t in tables:
    name = t[0]
    c.execute(f"SELECT COUNT(*) FROM [{name}]")
    print(f"  {name}: {c.fetchone()[0]} rows")

conn.close()
