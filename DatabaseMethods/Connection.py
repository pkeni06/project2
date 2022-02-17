import mysql.connector
from DatabaseMethods import config


class Database:
    def __init__(self):
        self.host = config.host
        self.password = config.password
        self.user = config.user
        self.database = config.database

    def getDB(self):
        try:
            db = mysql.connector.connect(host=self.host, user=self.user, password=self.password, database=self.database)
            return db
        except mysql.connector.errors.ProgrammingError as e:
            a = "1049 (42000): Unknown database '" + self.database + "'"
            if str(e) == a:
                conn = mysql.connector.connect(host=self.host, user=self.user, password=self.password)
                cursor = conn.cursor()
                cursor.execute("Create database " + self.database)

                db = mysql.connector.connect(host=self.host, user=self.user, password=self.password,
                                             database=self.database)
                return db
            else:
                return e
        except Exception as e:
            return e
