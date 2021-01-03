import os

def main():
    #Preprocessing
    os.system("python Preprocessing.py")

    #Train Model and pedict
    os.system("python Model.py")

if __name__ == '__main__':
    main()
