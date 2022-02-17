import mysql.connector.errors
from DatabaseMethods import Connection

"""def Authenticate(UserName, Password):
    conn = Connection.DatabaseMethods()
    db = conn.getDB()
    myCursor = db.cursor()

    pwd = str(hashlib.md5(Password.encode()).hexdigest())
    sql = "select * from users where UserName = %s and Password = %s"
    val = [UserName, pwd]
    myCursor.execute(sql, val)

    if not myCursor.fetchall():
        return False
    else:
        return True"""

def createTable():
    d = Connection.Database()
    db = d.getDB()
    myCursor = db.cursor()

    sql = "create table if not exists PatientPersonalDetails" \
          "(patientID INT Primary Key," \
          "Fname varchar(20)," \
          "Lname varchar(20)," \
          "Email varchar(50));"

    myCursor.execute(sql)
    db.commit()


def patientExists(patientID):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()

    sql = "select * from PatientPersonalDetails where patientID = '%s'" % patientID
    myCursor.execute(sql)

    if not myCursor.fetchall():
        return False
    else:
        return True


def deletePatient(patientID):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()
    if patientExists(patientID):
        sql = "Delete from PatientPersonalDetails where patientID = '%s'" % patientID
        myCursor.execute(sql)
        db.commit()
        if myCursor.rowcount > 0:
            return "Successful", 0
        else:
            return "Unsuccessful", 1
    else:
        return "patient doesn't exist", 1


def editPatient(PatientId, FirstName, LastName, Email):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()

    if patientExists(PatientId):
        sql = "update PatientPersonalDetails SET Fname = %s, Lname = %s, Email = %s where patientID = %s"
        values = [FirstName, LastName, Email, PatientId]
        myCursor.execute(sql, values)
        db.commit()
        return "Successful", 0
    else:
        return "Patient does not exist", 1


def addPatient(PatientID, FirstName, LastName, Email):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()
    print(PatientID, FirstName, LastName, Email)
    if checkIfEmailExists(Email):
        return "Email address already exists", 1

    values = [PatientID, FirstName, LastName, Email]
    sql = "Insert into PatientPersonalDetails(PatientID, Fname, Lname, Email) values (%s, %s, %s, %s)"
    try:
        myCursor.execute(sql, values)
    except mysql.connector.errors.IntegrityError as e:
        print(e)
        return "PatientID already exists", 1
    db.commit()
    return "Patient added successfully", 0


"""def checkIfUserNameExists(userName):
    conn = Connection.DatabaseMethods()
    db = conn.getDB()
    myCursor = db.cursor()

    sql = "select * from users where UserName = %s "
    val = [userName]
    myCursor.execute(sql, val)

    if not myCursor.fetchall():
        return False
    else:
        return True"""


def checkIfEmailExists(email):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()

    sql = "select * from PatientPersonalDetails where Email = %s "
    val = [email]
    myCursor.execute(sql, val)

    if not myCursor.fetchall():
        return False
    else:
        return True


def getPatient(PatientID):
    conn = Connection.Database()
    db = conn.getDB()
    myCursor = db.cursor()
    if patientExists(PatientID):
        sql = "select * from PatientPersonalDetails where PatientID = %s"
        val = [PatientID]
        myCursor.execute(sql, val)
        return myCursor.fetchall()[0], 0
    else:
        return "Patient Doesn't exist", 1


"""def getUser(userName):
    conn = Connection.DatabaseMethods()
    db = conn.getDB()
    myCursor = db.cursor()
    sql = "select * from patientdetails where userName = '%s'" % userName
    myCursor.execute(sql)
    return myCursor.fetchall()[0], 0"""

if __name__ == '__main__':
    createTable()
    print(addPatient("1", "Prathamesh", "Keni", "prathameshkeni6@gmail.com"))
    """print(addUser("2", "asdf", "ere", "aswe@gmail.com"))
    print(addUser("3", "okm", "poi", "aswe2@gmail.com"))"""
    #print(addPatient("1", "Prathamesh", "Keni", "prathameshkeni6@gmail.com"))
    """print(editPatient("1", "PK", "", "prathameshkeni6@gmail.com"))
    print(deletePatient("1"))
"""
