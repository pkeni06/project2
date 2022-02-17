import hashlib

from DatabaseMethods import Connection


def createTable():
    d = Connection.Database()
    db = d.getDB()
    myCursor = db.cursor()

    sql = "create table if not exists admin" \
          "(UserName varchar(200) Primary Key," \
          "Password varchar(200)," \
          "publicKey varchar(500));"

    myCursor.execute(sql)
    db.commit()

def Authenticate(UserName, Password):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()

    pwd = str(hashlib.md5(Password.encode()).hexdigest())
    sql = "select publicKey from admin where UserName = %s and Password = %s"
    val = [UserName, pwd]
    myCursor.execute(sql, val)

    if not myCursor.fetchall():
        return False
    else:
        return True


def Login(UserName, Password):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()

    pwd = str(hashlib.md5(Password.encode()).hexdigest())
    sql = "select publicKey from admin where UserName = %s and Password = %s"
    val = [UserName, pwd]
    myCursor.execute(sql, val)
    key = myCursor.fetchall()
    if not key:
        return "invalid username or password", 1
    else:
        return key, 0


def UsernameExists(username):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()
    sql = "select * from admin where UserName = %s"
    val = [username]
    myCursor.execute(sql, val)

    if not myCursor.fetchall():
        return False
    else:
        return True


def addAdmin(username, password, publicKey):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()
    if not UsernameExists(username):
        pwd = str(hashlib.md5(password.encode()).hexdigest())
        sql = "insert into admin values(%s, %s, %s)"
        data = [username, pwd, publicKey]
        myCursor.execute(sql, data)
        db.commit()
        return "admin added successfully", 0
    else:
        return "username exists", 1


def delAdmin(username, password):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()

    if Authenticate(username, password):
        sql = "delete from admin where username = %s"
        val = [username]
        myCursor.execute(sql, val)
        db.commit()
        return "Deleted successfully", 0
    else:
        return "username or password is wrong", 1


def getAdminByKey(publicKey):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()

    sql = "select * from admin where publickey = %s"
    val = [publicKey]

    myCursor.execute(sql, val)
    data = myCursor.fetchall()
    if data:
        return data[0], 0
    else:
        return "Invalid login credentials", 1


def changePassword(username, password, newPassword):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()

    if Authenticate(username, password):
        pwd = str(hashlib.md5(newPassword.encode()).hexdigest())
        sql = "update admin set password = %s where username = %s"
        val = [pwd, username]
        myCursor.execute(sql, val)
        db.commit()
        if myCursor.rowcount > 1:
            return "Changed successfully", 0
        else:
            return "Error occured", 1
    else:
        return "username or password is wrong", 1


if __name__ == "__main__":
    createTable()
    #print(Login("saish", "1234"))
