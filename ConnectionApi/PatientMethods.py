from flask_restful import Resource
from flask import request
from DatabaseMethods import Patient


class addPatient(Resource):
    def post(self):
        Pid = request.args.get("PatientID")
        Fname = request.args.get("FirstName")
        Lname = request.args.get("LastName")
        email = request.args.get("Email")
        response = Patient.addPatient(Pid, Fname, Lname, email)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class deletePatient(Resource):
    def delete(self):
        Pid = request.args.get("PatientID")
        response = Patient.deletePatient(Pid)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class editPatient(Resource):
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
    def get(self):
        Pid = request.args.get("PatientID")
        response = Patient.getPatient(Pid)

        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403
