from IFN import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

def main(how = 'binned'):
    # Load the Iris dataset
    iris = load_iris()
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['target'])

    if how == 'binned':
        # Convert numerical features to categorical by binning
        for col in iris['feature_names']:
            data[col] = pd.cut(data[col], bins=3, labels=["low", "medium", "high"])

    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Initialize and train the IFN model
    target_var = 'target'
    ifn_cat = IFN(train_data, target_var)
    ifn_cat.fit()
    ifn_cat.show()

    # Define the function for classification report
    def test_model(model, test_data):
        test_features = test_data.drop(columns=[target_var])
        test_labels = test_data[target_var].astype(int)

        predictions = model.predict(test_features)
        print(classification_report(test_labels, predictions))

    # Calculate min error probability
    print(f"Max error in the model based on Fano's Inequality: {ifn_cat.calculate_min_error_probability()}")

    # Use the function to get the classification report
    test_model(ifn_cat, test_data)

if __name__ == "__main__":
    # data_prep = 'binned'    # Replace with None if you want it as is
    data_prep = None    # Replace with binned if you want it ninned
    main(data_prep)