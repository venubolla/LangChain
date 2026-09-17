"""
Creates a local SQLite3 database file `student.db` with a STUDENT table,
populated with sample records.

Run this once before starting the Streamlit app:
    python sqlite.py
"""

import sqlite3

connection = sqlite3.connect("student.db")
cursor = connection.cursor()

# Create the STUDENT table
table_info = """
CREATE TABLE IF NOT EXISTS STUDENT (
    NAME VARCHAR(25),
    COURSE VARCHAR(25),
    SECTION VARCHAR(25),
    MARKS INT
)
"""
cursor.execute(table_info)

# Clear any existing rows so re-running this script doesn't duplicate data
cursor.execute("DELETE FROM STUDENT")

# Insert sample records
students = [
    ("Krish", "Data Science", "A", 90),
    ("John", "Data Science", "B", 100),
    ("Mukesh", "Data Science", "A", 86),
    ("Jacob", "DEVOPS", "A", 50),
    ("Dipesh", "DEVOPS", "A", 35),
]

cursor.executemany(
    "INSERT INTO STUDENT (NAME, COURSE, SECTION, MARKS) VALUES (?, ?, ?, ?)",
    students,
)

#executemany() takes the SQL template and the list, then loops through the list internally, 
#running the INSERT once per tuple, substituting each tuple's values into the ? placeholders in order

# Display inserted records for confirmation
print("The inserted records are:")
data = cursor.execute("SELECT * FROM STUDENT")
for row in data:
    print(row)


connection.commit()
connection.close()

print("\nstudent.db created successfully.")