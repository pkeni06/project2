import datetime
from functools import wraps
import uuid
import jwt
from flask import Flask, jsonify, request
from flask_restful import Api
from DatabaseMethods.Admin import getAdminByKey
from DatabaseMethods import Admin as admin
import DatabaseMethods.PatientMedicalDetails as PatientMedicalDetails
from flask_restful import Resource
from DatabaseMethods import Patient

app = Flask(__name__)
api = Api(app)
app.config['SECRET_KEY'] = 'tHi$_i$-tHe-$ecreT_kEy_for_JwT_TokeN'


def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):

        token = None

        if 'Authorization' in request.headers:
            token = request.headers['Authorization']

        # if 'x-access-tokens' in request.headers:
        #     token = request.headers['x-access-tokens']

        if not token:
            return jsonify({'message': 'a valid token is missing'})
        data = jwt.decode(token[7:], app.config['SECRET_KEY'], algorithms=["HS256"])
        data1 = getAdminByKey(data['public_id'])
        if data1[1] == 0:
            return f(data1[0][0])
        else:
            return data1[0], 403

    return decorator


class AdminLogin(Resource):
    def post(self):
        username = request.args.get("user_name")
        password = request.args.get("password")
        response = admin.Login(username, password)
        data = response[0][0][0]
        token = jwt.encode({'public_id': data, 'username': username,
                            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=90)},
                           app.config['SECRET_KEY'], algorithm="HS256")

        if response[1] == 0:
            return {'token': token}, 200
        else:
            return {'message': response[0]}, 403


class AddAdmin(Resource):
    def post(self):
        username = request.args.get("user_name")
        password = request.args.get("password")
        publicKey = str(uuid.uuid4())

        response = admin.addAdmin(username, password, publicKey)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class ChangeAdminPassword(Resource):
    @token_required
    def post(self):
        username = request.args.get("user_name")
        password = request.args.get("password")
        newPassword = request.args.get("new_password")
        response = admin.changePassword(username, password, newPassword)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class DeleteAdmin(Resource):
    @token_required
    def delete(self):
        username = request.args.get("user_name")
        password = request.args.get("password")

        response = admin.delAdmin(username, password)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class addPatient(Resource):
    @token_required
    def post(self):
        Pid = request.args.get("PatientID")
        Fname = request.args.get("FirstName")
        Lname = request.args.get("LastName")
        email = request.args.get("Email")
        print(Fname, Lname)
        response = Patient.addPatient(Pid, Fname, Lname, email)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class deletePatient(Resource):
    @token_required
    def delete(self):
        Pid = request.args.get("PatientID")
        response = Patient.deletePatient(Pid)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class editPatient(Resource):
    @token_required
    def post(self):
        Pid = request.args.get("PatientID")
        FName = request.args.get("FirstName")
        LName = request.args.get("LastName")
        Email = request.args.get("Email")
        response = Patient.editPatient(Pid, FName, LName, Email)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class getPatient(Resource):
    @token_required
    def get(self):
        Pid = request.args.get("PatientID")
        response = Patient.getPatient(Pid)

        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class addMedicalDetails(Resource):
    @token_required
    def post(self):
        id = request.args.get('PatientID')
        age = request.args.get('age')
        gender = request.args.get('Gender')
        height = request.args.get('height')
        weight = request.args.get('weight')
        ap_hi = request.args.get('ap_hi')
        ap_lo = request.args.get('ap_lo')
        cholestrol = request.args.get('cholestrol')
        gluc = request.args.get('gluc')
        smoke = request.args.get('smoke')
        alco = request.args.get('alco')
        active = request.args.get('active')
        bmi = request.args.get('bmi')

        response = PatientMedicalDetails.addRecord(id, age, gender, height, weight, ap_hi, ap_lo, cholestrol, gluc,
                                                   smoke, alco, active, bmi)
        print(response)

        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class getMedicalDetails(Resource):
    @token_required
    def get(self):
        Pid = request.args.get('PatientID')
        response = PatientMedicalDetails.getDetails(Pid)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class deleteMedicalDetails(Resource):
    @token_required
    def delete(self):
        Pid = request.args.get('PatientID')
        response = PatientMedicalDetails.deletePatient(Pid)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class predict(Resource):
    @token_required
    def post(self):
        Pid = request.args.get('PatientID')
        response = PatientMedicalDetails.addPrediction(Pid)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 404


api.add_resource(ChangeAdminPassword, '/changeAdminPassword')  # user_name, password, new_password
api.add_resource(AdminLogin, '/adminlogin')  # user_name, password
api.add_resource(DeleteAdmin, '/deleteAdmin')  # user_name, password
api.add_resource(AddAdmin, '/addAdmin')  # user_name, password

api.add_resource(addPatient, '/addPatient')  # PatientID, FirstName, LastName, Email
api.add_resource(editPatient, '/editPatient')  # PatientID, FirstName, LastName, Email
api.add_resource(deletePatient, '/deletePatient')  # PatientID
api.add_resource(getPatient, '/getPatient')  # PatientID

api.add_resource(addMedicalDetails, '/addMedicalDetails')
# PatientID, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi

api.add_resource(deleteMedicalDetails, '/delMedicalDetails')  # PatientID
api.add_resource(getMedicalDetails, '/getMedicalDetails')  # PatientID

api.add_resource(predict, '/predict')  # PatientID


def startServer(ip = "192.168.0.3"):
    app.run(host=ip)

if __name__ == '__main__':
    app.run(host="192.168.0.3")
