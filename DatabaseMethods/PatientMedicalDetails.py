import DatabaseMethods.Connection as connection
from MachineLearning.TestModel import PredictResult as predict


def createTable():
    d = connection.Database()
    db = d.getDB()
    myCursor = db.cursor()

    sql = "create table if not exists PatientDetails" \
          "(id INT Primary Key," \
          "age INT," \
          "gender INT," \
          "height FLOAT," \
          "weight FLOAT," \
          "ap_hi FLOAT," \
          "ap_lo FLOAT," \
          "cholesterol FLOAT," \
          "gluc FLOAT," \
          "smoke INT," \
          "alco INT," \
          "active INT," \
          "bmi FLOAT," \
          "cardio FLOAT null);"

    myCursor.execute(sql)
    db.commit()


def AccountExists(userID):
    d = connection.Database()
    db = d.getDB()
    myCursor = db.cursor()

    sql = "select * from patientpersonaldetails where patientID = %s"
    val = [userID]
    myCursor.execute(sql, val)

    if not myCursor.fetchall():
        return False
    else:
        return True


def PatientExists(PatientID):
    d = connection.Database()
    db = d.getDB()
    myCursor = db.cursor()

    sql = "select * from patientdetails where id = %s"
    val = [PatientID]
    myCursor.execute(sql, val)

    if not myCursor.fetchall():
        return False
    else:
        return True


def addRecord(id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi):
    d = connection.Database()
    db = d.getDB()
    myCursor = db.cursor()

    sql = "insert into PatientDetails (id, age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco," \
          "active,bmi) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    values = [id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi]

    if AccountExists(id):
        if PatientExists(id):
            return "Record already exist", 1
        else:
            try:
                myCursor.execute(sql, values)
                db.commit()
                return "Details added successfully", 0
            except Exception as e:
                return e, 1
    else:
        return "User account does not exist", 1


def addPrediction(id):
    d = connection.Database()
    db = d.getDB()
    myCursor = db.cursor()
    prediction = predict(getDetails(id))
    sql = "update PatientDetails set cardio = %s where id = %s"
    values = [prediction, id]
    if PatientExists(id):
        myCursor.execute(sql, values)
        db.commit()
        return prediction, 0
    else:
        return "patient doesn't exist", 1


def getDetails(id):
    d = connection.Database()
    db = d.getDB()
    myCursor = db.cursor()

    if PatientExists(id):
        sql = "select * from patientDetails where id = %s "
        val = [id]
        myCursor.execute(sql, val)
        return myCursor.fetchall()[0], 0
    else:
        return "Patient doesn't exist", 1


def deletePatient(id):
    d = connection.Database()
    db = d.getDB()
    myCursor = db.cursor()

    if PatientExists(id):
        sql = "delete from patientDetails where id = %s "
        val = [id, ]
        myCursor.execute(sql, val)
        db.commit()
        return myCursor.rowcount, 0
    else:
        return "Patient doesn't exist", 1


if __name__ == "__main__":
    createTable()
    # addRecord(1,16568,1,170,68.0,140,70,1,1,0,0,0,23.529411764705884)
    # deletePatient(1)
    # print((getDetails(1)))
    # addRecord(2,58,1,5.135798437050262,4.219507705176107,130,90,3,1,0,0,1,23.529411764705873)
    # addRecord(3,42,1,5.198497031265826,4.653960350157523,132,92,2,3,1,0,1,32.05030371478279)
    # pass
