from functools import wraps

import jwt
from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
from AdminMethods import *
from PatientMethods import *
from MedicalDetailsMethods import *
from DatabaseMethods.Admin import getAdminByKey
app = Flask(__name__)
api = Api(app)


api.add_resource(ChangeAdminPassword, '/changeAdminPassword')
api.add_resource(AdminLogin, '/adminlogin')
api.add_resource(DeleteAdmin, '/deleteAdmin')

api.add_resource(addPatient, '/addPatient')
api.add_resource(editPatient, '/editPatient')
api.add_resource(deletePatient, '/deletePatient')
api.add_resource(getPatient, '/getPatient')

api.add_resource(addMedicalDetails, '/addMedicalDetails')
api.add_resource(deleteMedicalDetails, '/delMedicalDetails')
api.add_resource(getMedicalDetails, '/getMedicalDetails')

if __name__ == '__main__':
    app.run(host="127.0.0.1")

