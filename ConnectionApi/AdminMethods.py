import uuid

from flask_restful import Resource
from flask import request
from DatabaseMethods import Admin as admin

class AdminLogin(Resource):
    def post(self):
        username = request.args.get("user_name")
        password = request.args.get("password")
        if admin.Authenticate(username, password):
            response = {'message': 'Login Successful'}, 200
        else:
            response = {'message': 'Login Unsuccessful'}, 403
        return response


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
    def delete(self):
        username = request.args.get("user_name")
        password = request.args.get("password")

        response = admin.delAdmin(username, password)
        if response[1] == 0:
            return {'message': response[0]}, 200
        else:
            return {'message': response[0]}, 403
