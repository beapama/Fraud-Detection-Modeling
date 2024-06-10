import pandas as pd 

validation_data = pd.read_csv("fraud-detection-evaluate-model/assets/validation_data.csv")

print(validation_data.head())

actual, predicted = validation_data['actual'], validation_data['predicted']

print(actual.head())

print(predicted.head())

print("THE ACTUAL AND PREDICTED VALUES HAVE BEEN READ INTO A DATAFRAME")

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

precision = precision_score(actual, predicted)

print("precision: ", precision)


accuracy = accuracy_score(actual, predicted)

print("accuracy: ", accuracy)

recall = recall_score(actual, predicted)

print("recall: ", recall)

f1_score = f1_score(actual, predicted)

print("f1_score: ", f1_score)

import pickle

model_filename = 'fraud-detection-evaluate-model/assets/random_forest_model.pkl'
with open(model_filename, 'rb') as model_file:
    loaded_rf_model = pickle.load(model_file)


importances = loaded_rf_model.feature_importances_
features = loaded_rf_model.feature_names_in_

feature_importance_df = pd.DataFrame({"features":features, "importances": importances})
feature_importance_df.sort_values("importances", ascending=False)

print(feature_importance_df)

feature_importance_df.to_csv("feature_importance.csv", index=False)