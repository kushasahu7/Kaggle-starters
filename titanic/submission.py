import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier


features = ["Sex", "Pclass", "SibSp", "Parch", "Age", "Embarked"]
# for dirname, _, filenames in os.walk('../titanic/data'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

def get_training_data():
    train_data = pd.read_csv("../titanic/data/train.csv")
    return train_data

def get_test_data():
    test_data = pd.read_csv("../titanic/data/test.csv")
    return test_data

def divide_data(train_data, test_data):
    y = train_data["Survived"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])
    return y, X, X_test


def get_model():
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    return model

def train(model, X, y):
    model.fit(X, y)

def predict(model, X_test, test_data, submissions_file_name):
    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv(submissions_file_name, index=False)


def run():
    train_data = get_training_data()
    test_data = get_test_data()
    y, X, X_test = divide_data(train_data, test_data)
    model = get_model()
    train(model, X, y)
    predict(model, X_test, test_data, 'second_submission.csv')
print("Your submission was successfully saved!")
run()
