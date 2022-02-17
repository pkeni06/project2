import mysql.connector

db = mysql.connector.connect(host="127.0.0.1", user="root1", password="root", database="project")
print(db)