from flask_restful import Resource
from flask import request
from DatabaseMethods import PatientMedicalDetails


class addMedicalDetails(Resource):
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

        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class getMedicalDetails(Resource):
    def get(self):
        Pid = request.args.get('PatientID')
        response = MedicalDetails.getPatient(Pid)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403


class deleteMedicalDetails(Resource):
    def delete(self):
        Pid = request.args.get('PatientID')
        response = MedicalDetails.delPatient(Pid)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403
