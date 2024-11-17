from IFN import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Define the function for classification report
def test_model(model, test_data, target_var):
    test_features = test_data.drop(columns=[target_var])
    test_labels = test_data[target_var]

    predictions = model.predict(test_features)
    print(classification_report(test_labels, predictions))


def test_with_pickle(test_data, target_var, model_path = 'ifn_model.pkl'):
    # Load the model
    loaded_model = IFN.load(model_path)
    
    # Test the loaded model
    test_model(loaded_model, test_data, target_var)
    return loaded_model

def main():
    # Load the dataset
    iris = load_iris()
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                    columns= iris['feature_names'] + ['target'])

    # Map target values to species names
    target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    data['target'] = data['target'].map(target_names)
    
    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Initialize and train the IFN model
    target_var = 'target'
    model_path = 'ifn_model.pkl'
    ifn_cat = IFN()
    ifn_cat.fit(train_data, target_var)
    ifn_cat.save(model_path)

    # Calculate min error probability
    print(f"Max error in the model based on Fano's Inequality: {ifn_cat.calculate_min_error_probability()}")
    
    # Test model's saving and loading ability, along with performance (get the classification report)
    model = test_with_pickle(test_data, target_var, model_path)
    model.show()

if __name__ == "__main__":
    main()