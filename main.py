from ConnectionApi import FlaskMain as Flask
from MachineLearning.TrainModel import trainModel as train
if __name__ == "__main__":
    i = 10
    while(i != -1):
        print("Enter Choice:")
        print("1. Start Server")
        print("2. Train Model")
        print("3. Create tables")
        print("4. Exit")
        i = int(input())

        if i == 1:
            Flask.startServer()
        elif i == 2:
            train()
        elif i == 3:
            pass
        elif i == 4:
            exit(0)
        else:
            print("Invalid choice, Try again")



