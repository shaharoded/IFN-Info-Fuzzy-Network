# Information Fuzzy Network (IFN) Model

The Information Fuzzy Network (IFN) model is designed to find the best attribute to split by at each level of the model. It works with both categorical and numeric attributes, ensuring that the splits maximize the mutual information between the input variables and the target variable. The model also implements Fano's Inequality to estimate the minimum prediction error.
The model allows for 1 attribute on each level, meaning all nodes will be splitted by it, if significant for them, or will be directed to the target nodes ('leaf' nodes) and be determined as terminal.

![Network Visualization](IFN_Image.png)


## Key Features

1. **General Attribute Splitting**: The model handles both categorical and numeric attributes, finding the best attribute to split by at each level.
2. **Recursive Binning for Numeric Values**: For numeric attributes, the model recursively finds the best splits by evaluating the significance of each potential split.
3. **Significance Testing**: Uses a significance test to determine whether a split is meaningful and should be included in the final model.
4. **Fano's Inequality**: Implements Fano's Inequality to estimate the minimum prediction error.

## How It Works

### Initialization

The `IFN` class initializes with the training data, target variable, and a significance threshold. During initialization, it calls the `init_create_bin_suggestions` method to determine the optimal binning thresholds for each numeric attribute, if exists.

### Finding Best Attribute to Split

The model finds the best attribute to split by evaluating the mutual information gain for each attribute. It selects the attribute that maximizes the significance score (using chi2 test) and uses it to split the data at each level of the model. If a split is not significant, test will return a score of 0.

### Recursive Splitting for Numeric Attributes

For numeric attributes, the model recursively finds the best splits by:
1. Identifying the unique values of the attribute.
2. Testing different thresholds to see how they split the data.
3. Ensuring the splits are meaningful by evaluating the significance using `_significance_test`.

## Available Public Methods

### IFN Class

- **`__init__(self, train_data, target, P_VALUE_THRESH, max_depth, weights_type)`**: Initializes the IFN model with the training data, target variable, and significance threshold. You can limit the depth to avoid overfitting (on edge cases) and weights_type controls the weights displayed when plotting.
- **`plot_network(self)`**: Plots the network visualization of the IFN model using NetworkX.
- **`predict(self, df)`**: Predicts the target variable for a given dataframe using the trained IFN model.
- **`calculate_min_error_probability(self)`**: Assess the max error from the model, using the edge weight.

You can see a testing sample for both ordinal and categorical features on the Testing section of the notebook.