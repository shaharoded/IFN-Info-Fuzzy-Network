# Information Fuzzy Network (IFN) Model

The Information Fuzzy Network (IFN) machine learning model is designed to statistically find the best attribute to split by at each level of the model using chi-square test. It works with both categorical and numeric attributes, ensuring that the splits maximize the mutual information between the input variables and the target variable, while, unlike decision trees, split only when the improvement is significant. The model also implements Fano's Inequality to estimate the maximum prediction accuracy (min error) using the mutual information theory. Each level of the model allows for one attribute split, meaning all nodes at that level will be split by the selected attribute if significant, or directed to the target nodes ('leaf' nodes) and determined as terminal.

As shown in the viz, the built model resembles a "fully connected tree", and will choose the predicted class based on the majority rule at the terminal node.

![Network Visualization](Images/IFN_Iris.png)

The IFN model is useful for classification tasks usually handled with decision trees. It is a statisticaly stable model which performs feature selection in a built-in method, using chi test. 

Below are the evaluation results on the Iris dataset, demonstrating good prediction performance.

![Network Visualization](Images/IFN_Classification_Report.png)

Train and prediction are possible on both categorical and numeric attributes, but note that the recursive optimal split detection on numeric ordinal variables can increase the runtime significantly, depends on the size of the data.

Target column can be either int or str, as long as it's descrete and has a reasonable amount of unique labels compared to the data size.

To start, run the following command in the terminal:

```
pip install -r requirements.txt
python main.py
```

## Key Features

1. **General Attribute Splitting**: Handles both categorical and numeric attributes, identifying the best attribute to split by at each level.
2. **Recursive Binning for Numeric Values**: Recursively finds the best splits for numeric attributes by evaluating the significance of each potential split.
3. **Significance Testing**: Uses chi-square tests to determine whether a split is meaningful and should be included in the final model.
4. **Fano's Inequality**: Implements Fano's Inequality to estimate the minimum prediction error.

## How It Works

### Initialization

The `IFN` class initializes with the training data, target variable, and a significance threshold. During initialization, it calls the `__init_create_bin_suggestions` method to determine the optimal binning thresholds for each numeric attribute, if any exist.

### Finding Best Attribute to Split

The model finds the best attribute to split by evaluating the mutual information gain for each attribute. It selects the attribute that maximizes the significance score (using the chi-square test) and uses it to split the data at each level. If a split is not significant, the test returns a score of 0.

### Recursive Splitting for Numeric Attributes

For numeric attributes, the model recursively finds the best splits by:
1. Identifying the unique values of the attribute.
2. Testing different thresholds to see how they split the data.
3. Ensuring the splits are meaningful by evaluating the significance using the `__significance_test` method.

## Available Public Methods

### IFN Class Methods

- **`__init__(self)`**: Initializes the IFN model's object.
- **`fit(self, train_data, target, P_VALUE_THRESH, max_depth, weights_type)`**: Building the network with the training data, target variable, and significance threshold, while performing all necessary calculations. You can limit the depth to avoid overfitting (on edge cases) and weights_type controls the weights displayed when plotting.
- **`show(self)`**: Plots the network visualization of the IFN model using NetworkX. NOTE: The plot represent "stronger" edges (higher probability) using darker colors, which should represent how a prediction will be made if a records "landed" in that node.
- **`predict(self, df)`**: Predicts the target variable for a given dataframe using the trained IFN model.
- **`calculate_min_error_probability(self)`**: Assess the min error from the model on unseen data, using the edge weight. To use properly, train the model on your full dataset and and call this method. This will provide an upper boundry to the model's accuracy on unseen data.
- **`save(self, file_path)`**: Save the model to a pickle file, at the designated file_path.
- **`load(file_path)`**: Loads a pre-trained model from file_path. Activation example can be shown at `main.py`.

## Getting Started

### Using PYPI
The Info-Fuzzy Network (IFN) model is now available on PyPI. You can install it directly using pip:

```bash
pip install InfoFuzzyNetwork --upgrade
from InfoFuzzyNetwork import IFN
```

Be sure to have installed version 1.0.4. You can view using:

```bash
!pip show InfoFuzzyNetwork
```

### Or Use It Locally, Using Git
1. Clone this repository. Run the following command:

```bash
git clone https://github.com/shaharoded/IFN-Info-Fuzzy-Network.git
```

2. Set up a virtual environment and activate it:

```bash
python -m venv venv
.\venv\Scripts\Activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Use the `main.py` to load, train and test an instance of the model, as well as a reference for mathods usage.

```bash
python main.py
```

NOTE: The model will log it's progress and selected features along the training process.

## GitHub Push Actions
To commit and push all changes to the repository follow these steps:

    ```bash
    git add .
    git commit -m "Reasons for disrupting GIT (commit message)"
    git push -u origin main / git push -f origin main   # If you want to force push
    ```

## Acknowledgments
This implementation was inspired by the `Advanced Topics in ML` course taught by **professor M.Last** at BGU. The model has been implemented in Python as a personal challenge and for educational and research purposes. It is shared under an open-source license for anyone interested in using or extending it.
While the code is entirely original and written by me, the algorithm and theoretical concepts are credited to M. Last (and any associates of his). 

Relevant foundational articles include:

 - Maimon, O., Last, M. (2001). Information-Theoretic Connectionist Networks. In: Knowledge Discovery and Data Mining. Massive Computing, vol 1. Springer, Boston, MA. https://doi.org/10.1007/978-1-4757-3296-2_3
 - Last, Mark, Abraham Kandel, and Oded Maimon. "Information-theoretic algorithm for feature selection." Pattern Recognition Letters 22.6-7 (2001): 799-811.‏
 - Last, Mark, and Menahem Friedman. "Black-box testing with info-fuzzy networks." Artificial Intelligence Methods in Software Testing. 2004. 21-50.‏


## Disclaimer
This project implements an Info-Fuzzy Network (IFN) model based on the foundational concepts described in the work of M. Last and his associates. This implementation is independent, original, and created for educational and research purposes. Any resemblance to proprietary software is coincidental.

## TO-DO
A discrepancy has been observed between local and cloud environments when plotting. In cloud environments, loading the model can occasionally trigger an unintended `self.show()` call. Stricter QA and validation across different use cases could address this issue, though it does not significantly impact regular functionality.
It is also important to note that plotting dimensions might be interpreted differently across various environments, so for the best experience you may prefer to use this code locally.