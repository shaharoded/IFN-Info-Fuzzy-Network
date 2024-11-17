import numpy as np
import pandas as pd
import math
from scipy.stats import chi2
from scipy.optimize import fsolve
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
from functools import wraps
import time
import pickle


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_input_type(expected_type):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip the first argument which is 'self' in instance methods
            for arg in args[1:]:
                if not isinstance(arg, expected_type):
                    raise TypeError(f"Argument must be of type {expected_type}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def time_method_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[Runtime Info]: Execution of {func.__name__} function took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


class IFN:
    '''
    IFN model implementation.
    This implementation can only handle discrete target columns.
    The ordinal columns binarization is a simplified version, deciding on the optimal bins per column
    before starting to build the network.
    '''
    class Node:
        '''
        Node class for the IFN model.
        '''
        def __init__(self, data: pd.DataFrame, level_attr: str = 'root', level_attr_val: str = None, is_terminal: bool = False, color: str = 'blue'):
            '''
            Initialize the node.
            Args:
                data: pandas DataFrame, the subset of the data at node Z.
                level_attr: The name of the feature on that node's level.
                level_attr_val: The value of that node in level_attr.
                is_terminal: bool, whether the node is a terminal node.
                color: str, the color of the node.
            '''
            self.data = data
            self.level_attr = level_attr
            self.level_attr_val = level_attr_val
            self.split_attr = None
            self.children = []
            self.is_terminal = is_terminal
            self.color = color


    def __init__(self):
        self.train_data = None  # Training DataFrame (including target column)
        self.target = None  # Target column name (str)
        self.P_VALUE_THRESH = None  # Statistical thresh for split by the model
        self.input_vars = {}  # Input vars with a mark of 1 for 'used' on 0 for not used
        self.N = 0  # Number of records in the trained data
        self.bin_suggestions = None     # Split points suggestions for every feature in the data
        self.root = None    # The root's node
        self.target_nodes = []  # The target nodes, based on the unique values of the target column
        self.nodes = [self.root] + self.target_nodes    # All model's nodes
        self.levels = []    # Levels are groups of nodes from the same layer (feature) in the model
        self.edges = []  # A tuple like (pointer, poitee, {weight, probability}), all non-terminal nodes get weight == None
        self.max_depth = 0  # Max possible selected layers for the model
        self.weights_type = ''  # 'mutual' or 'probability', for plotting only.
        self.plot = None    # The plot object (instead of 'print')
        self.ready = False    # A readiness flag, signaling other methods from the model can be activated


    def __init_target_nodes(self):
        '''
        Initialize the target nodes based on the unique values in the target variable.
        '''
        target_values = self.train_data[self.target].unique()
        target_nodes = [IFN.Node(self.train_data[self.train_data[self.target] == val], level_attr=self.target, level_attr_val=val, is_terminal=True, color='red') for val in target_values]
        for target_node in target_nodes:
            logging.info(f'Target Class Initiated: "{target_node.level_attr_val}", No. Records = {len(target_node.data)}')
        return target_nodes


    def __init_create_bin_suggestions(self):
        '''
        Return value is a dictionary of suggestions for binning per column, if column
        type is 'ordinal'.
        {column_name: [threshold_val_1, threshold_val_2,...]}
        '''
        def find_best_threshold(data: pd.DataFrame, col: str):
            """
            Find the best threshold for a given column in the data.

            Args:
                data: pandas DataFrame, the dataset.
                col: string, the column name.

            Returns:
                The best threshold value and its significance score.
            """
            unique_vals = sorted(data[col].unique())
            best_threshold = None
            max_significance = 0

            for T in unique_vals:
                bin_column = data[col].apply(lambda x: 0 if x < T else 1)
                temp_data = data.copy()
                temp_data[col] = bin_column
                significance = self.__significance_test(temp_data, col)
                if significance > max_significance:
                    max_significance = significance
                    best_threshold = T

            return best_threshold, max_significance

        def recursive_find_splits(data: pd.DataFrame, col: str, thresholds: list = None):
            """
            Recursively find the best splits for a given column in the data.

            Args:
                data: pandas DataFrame, the dataset.
                col: string, the column name.
                thresholds: list, the list of thresholds (default is None).

            Returns:
                List of thresholds.
            """
            if thresholds is None:
                thresholds = []

            best_threshold, max_significance = find_best_threshold(data, col)

            if best_threshold is not None and max_significance > 0:
                thresholds.append(best_threshold)
                logging.info(f"Found significant split at {best_threshold} with significance {max_significance}")

                # Split the data into two subsets
                left_data = data[data[col] < best_threshold]
                right_data = data[data[col] >= best_threshold]

                # Recursively find splits in each subset
                if not left_data.empty:
                    recursive_find_splits(left_data, col, thresholds)
                if not right_data.empty:
                    recursive_find_splits(right_data, col, thresholds)

            return sorted(thresholds)

        numerical_attributes = [attr for attr in self.train_data.columns if attr != self.target and pd.api.types.is_numeric_dtype(self.train_data[attr])]
        if not numerical_attributes:
            return {}

        logging.info('Looking for optimal threshold points in numerical columns')
        bin_suggestions = {}

        for attr in numerical_attributes:
            logging.info(f"Processing column: {attr}")
            thresholds = recursive_find_splits(self.train_data, attr)
            if thresholds:
                bin_suggestions[attr] = thresholds

        # Create threshold per level if attr is ordinal
        # bin_suggestions[attr] is the bins edges for the attribute
        for attr in bin_suggestions:
            bin_suggestions[attr] = [-float('inf')] + bin_suggestions[attr] + [float('inf')]
            bin_labels = [f"({bin_suggestions[attr][i]}, {bin_suggestions[attr][i+1]}]" for i in range(len(bin_suggestions[attr])-1)]
            binned_column = pd.cut(self.train_data[attr], bins=bin_suggestions[attr], labels=bin_labels)
            self.train_data[attr] = binned_column
        return bin_suggestions      # Replace init dictionary


    def __mutual_information(self, data: pd.DataFrame, attr: str) -> float:
        '''
        Mutual information calculation for the IFN model, calculated as a sum of all
        conditional mutual informations per xi in X (input variables) compared to Y (target variable).

        Args:
            data: the subset of the data at node Z.
            attr: string, the input variable (column name) for which mutual information is calculated.

        Returns:
            float: The mutual information value.
        '''
        Z = len(data)
        mi = 0
        for yi in data[self.target].unique():
            for xi in data[attr].unique():
                P_Yj_Xi = len(data[(data[attr] == xi) & (data[self.target] == yi)]) / self.N
                P_Yj_Xi_Z = len(data[(data[attr] == xi) & (data[self.target] == yi)]) / Z
                P_Xi_Z = len(data[data[attr] == xi]) / Z
                P_Yj_Z = len(data[data[self.target] == yi]) / Z

                if P_Yj_Xi == 0 or P_Xi_Z == 0 or P_Yj_Z == 0:
                    mci = 0
                else:
                    mci = P_Yj_Xi * np.log2(P_Yj_Xi_Z / (P_Xi_Z * P_Yj_Z))

                mi += mci
        return mi


    def __significance_test(self, data: pd.DataFrame, attr: str) -> float:
        '''
        Perform a significance test using chi2 test.

        Args:
            data: pandas DataFrame, the subset of the data at node Z.
            attr: string, the attribute to test.

        Returns:
            float: The mutual information value if significant, else 0.
        '''
        mi = self.__mutual_information(data, attr)
        statistic = 2 * math.log(2) * self.N * mi

        k = len(data[attr].unique())
        m = len(data[self.target].unique())
        dof = (k - 1) * (m - 1)

        critical_value = chi2.ppf(1 - self.P_VALUE_THRESH, dof)
        return mi if statistic >= critical_value else 0


    def __best_split(self, level: list) -> str:
        '''
        Return the next attribute to split by.

        Args:
            level: The nodes from the previous split. A list of nodes.

        Returns:
            string: the name of the best variable to split on.
        '''
        available_attributes = [attr for attr in self.input_vars if self.input_vars[attr] == 0]
        max_mi = 0
        best_attr = None
        terminal_nodes = []
        for attr in available_attributes:
            mi_0_nodes = []  # Nodes that will turn terminal if this attr is chosen
            mi = 0
            for node in level:
                # Calculate MI
                node_mi = self.__significance_test(node.data, attr)
                mi_0_nodes.append(node) if node_mi == 0 else None
                mi += node_mi
            if mi > max_mi:
                max_mi = mi
                best_attr = attr
                terminal_nodes = mi_0_nodes

        # Mark the best attribute as used
        if not best_attr or self.max_depth == 0:
            terminal_nodes = level
        else:
            self.input_vars[best_attr] = 1  # Mark the best attribute as used
            if isinstance(self.max_depth, int):
                self.max_depth -= 1

        # Connect terminal nodes to target nodes
        for node in terminal_nodes:
            node.is_terminal = True
            for target_node in self.target_nodes:
                node.children.append(target_node)
                w = self.__edge_weight(node, target_node)
                self.edges.append((node, target_node, w))

        return best_attr


    def __edge_weight(self, node: 'IFN.Node', target_node: 'IFN.Node') -> dict:
        '''
        Calculates the weight of an edge from a terminal node to a target node.
        Avoid division by 0 with small values as default (np.finfo(float).eps).

        Args:
            node: IFN.Node, the terminal node.
            target_node: IFN.Node, the target node.

        Returns:
            dict: A dictionary with 'weight' and 'probability' keys.
        '''
        node_n_of_records = max(len(node.data), np.finfo(float).eps)
        mutual_n_of_records = max(len(node.data[node.data[self.target] == target_node.level_attr_val]), np.finfo(float).eps)
        target_n_of_records = max(len(target_node.data), np.finfo(float).eps)

        P_Y = target_n_of_records / self.N
        P_Y_given_node = mutual_n_of_records / node_n_of_records
        P_mutual = mutual_n_of_records / self.N
        return {'weight': P_mutual * np.log2(P_Y_given_node / P_Y),
                'probability': P_Y_given_node}


    def __build_network(self, level: list, level_name: str = 'root'):
        '''
        Build the network recursively.

        Args:
            level: The nodes from the previous split. A list of nodes.
            level_name: The name of the level, the previous 'best attr'.
        '''
        if not level:
            return

        self.levels.append(level)
        # Choose the best attr for all level nodes
        best_attr = self.__best_split(level)
        logging.info(f'Current Level: {level_name}, Best Split: {best_attr}')
        if best_attr is None:
            logging.info('No attribute was chosen, define level nodes as terminal and exit function')
            self.levels.append(self.target_nodes)
            return

        next_level = []
        level_name = best_attr
        level_data = pd.concat([node.data for node in level if not node.is_terminal])
        for value in level_data[best_attr].unique():
            subset = level_data[level_data[best_attr] == value]
            child_node = IFN.Node(subset, level_attr=best_attr, level_attr_val=value)
            self.nodes.append(child_node)
            next_level.append(child_node)
            for node in level:
                if not node.is_terminal:
                    node.split_attr = best_attr
                    self.edges.append((node, child_node, None))
                    node.children.append(child_node)
        self.__build_network(next_level, level_name)


    @time_method_call
    def fit(self, train_data: pd.DataFrame, target: str, P_VALUE_THRESH: float = 0.1, max_depth: int = None, weights_type: str = 'probability'):
        '''
        Fit the IFN model to the training data, based on the target variable.
        Args:
            train_data (pd.DataFrame): The training data (including the target column)
            target (str): The target column name in the train_data
            P_VALUE_THRESH (float):
            max_depth (int):
            weights_type (str): 
        '''
        # Update init variables
        self.train_data = train_data
        self.target = target
        self.P_VALUE_THRESH = P_VALUE_THRESH
        self.input_vars = {col: 0 for col in train_data.columns if col != target}  # Input vars with 1 for 'used'
        self.N = len(train_data)
        self.root = IFN.Node(train_data, level_attr='root', color='purple')
        self.target_nodes = self.__init_target_nodes()
        self.nodes = [self.root] + self.target_nodes
        self.max_depth = max_depth
        self.weights_type = weights_type
        
        # Train the model and build the network
        self.bin_suggestions = self.__init_create_bin_suggestions()  # A simplified version of threshold optimization
        self.__build_network([self.root])
        # Clean unused numerical column names
        self.bin_suggestions = {column: self.bin_suggestions[column] for column in self.bin_suggestions if self.input_vars[column] == 1}
        self.plot = self.__plot_network()
        
        # Flag the model is good-to-go
        self.ready = True

    
    def __plot_network(self):
        '''
        Plot the IFN network and return the Figure object.
        Ensures that only one figure is active at a time.
        '''
        fig = plt.figure(figsize=(13, 5))  # Create a new figure
        G = nx.DiGraph()
        colors = []
        pos = {}
        labels = {}
        edge_weights = []

        terminal_nodes = set()

        # Add nodes to the graph
        for level_index, level in enumerate(self.levels):
            for node_index, node in enumerate(level):
                node_id = id(node)
                G.add_node(node_id)
                colors.append(node.color if node.color != 'purple' else 'mediumpurple')
                pos[node_id] = (node_index, -level_index)
                if node.color != 'purple':
                    labels[node_id] = node.level_attr_val
                if not any(child for (parent, child, _) in self.edges if id(parent) == node_id):
                    terminal_nodes.add(node_id)

        for (parent, child, weight) in self.edges:
            parent_id = id(parent)
            child_id = id(child)
            if weight is not None:
                edge_weights.append(weight['probability'] if self.weights_type == 'probability' else weight['weight'])
                G.add_edge(parent_id, child_id)
            else:
                G.add_edge(parent_id, child_id)
                edge_weights.append(0.0)

        norm = mcolors.Normalize(vmin=-0.5, vmax=max(edge_weights))
        cmap = plt.cm.get_cmap('Purples')
        edge_colors = []
        for (parent, child) in G.edges():
            if child in terminal_nodes:
                weight = edge_weights.pop(0)
                edge_colors.append(cmap(norm(weight)))
            else:
                edge_colors.append(cmap(0.4))

        nx.draw(G, pos, with_labels=False, node_color=colors, node_size=800, alpha=0.6, edge_color=edge_colors)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=5, font_color='black', font_family='sans-serif', font_weight='bold')

        norm = mcolors.Normalize(vmin=0, vmax=max(edge_weights))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Edge Weight')

        max_x = max(x for x, y in pos.values())
        for level_index, level in enumerate(self.levels):
            if level:
                first_node = level[0]
                first_node_pos = pos[id(first_node)]
                level_name = first_node.level_attr
                plt.plot([0, max_x + 0.1], [first_node_pos[1], first_node_pos[1]], color='gray', linestyle='--', linewidth=0.5)
                plt.text(max_x + 0.15, first_node_pos[1], level_name, verticalalignment='center', fontsize=9, color='gray')

        return fig  # Return the Figure object

    
    def show(self):
        '''
        Display the plot of the IFN network.
        '''
        if not self.ready:
            raise RuntimeError("Cannot use show() on an untrained model. Call fit() first.")
        self.plot.show()


    def calculate_min_error_probability(self) -> float:
        '''
        Calculate the minimum prediction error P_e based on Fano's Inequality using the complete feature set.
        NOTE: This method allows you to train the model on the full data (no test unseen data), and get an approximation
        of the max error from the model.
        
        Returns:
            float, the minimum prediction error P_e.
        '''
        if not self.ready:
            raise RuntimeError("Cannot use calculate_min_error_probability() on an untrained model. Call fit() first.")
        
        # Calculate the entropy H(A) of the target attribute
        prob = self.train_data[self.target].value_counts(normalize=True)
        entropy = -np.sum(prob * np.log2(prob))

        # Calculate the mutual information MI(A;I) as the sum of weights across all edges in the IFN
        mi = sum(weight['weight'] for _, _, weight in self.edges if weight is not None and isinstance(weight['weight'], (float, int)) and not np.isnan(weight['weight']))

        # Calculate the conditional entropy H(A|I)
        h_conditional = entropy - mi

        # Determine the number of classes M in the target attribute
        num_classes = self.train_data[self.target].nunique()

        # Define the Fano's inequality function to solve for P_e
        def fano_inequality(P_e):
            return h_conditional - (-P_e * np.log2(P_e) - (1 - P_e) * np.log2(1 - P_e) + P_e * np.log2(num_classes - 1))

        # Try multiple initial guesses for P_e
        initial_guesses = [0.1, 0.5, 0.9]
        for guess in initial_guesses:
            try:
                min_error_probability = fsolve(fano_inequality, guess)[0]
                if 0 <= min_error_probability <= 1:
                    return min_error_probability
                else:
                    logging.warning(f"Invalid result with initial guess {guess}: {min_error_probability}")
            except Exception as e:
                logging.error(f"Error with initial guess {guess}: {e}")

        # If all guesses fail, return a default value or handle the error appropriately
        logging.warning('Failed to converge with all initial guesses.')
        return None


    def __predict_record(self, record: pd.Series):
        '''
        Predict the target value for a single record.

        Args:
            record: dict, a single record with feature values.

        Returns:
            The predicted target value.
        '''
        current_node = self.root
        while not current_node.is_terminal:
            split_attr = current_node.split_attr
            split_value = record.get(split_attr)
            for child in current_node.children:
                if child.level_attr_val == split_value:
                    current_node = child
                    break

        # If current_node is terminal, find the child with the maximum edge weight
        if current_node.is_terminal:
            best_child = max(current_node.children, key=lambda x: next((weight['probability'] for parent, child, weight in self.edges if parent == current_node and child == x), 0))
            return best_child.level_attr_val

        return None


    @validate_input_type(pd.DataFrame)
    @time_method_call
    def predict(self, df: pd.DataFrame) -> pd.Series:
        '''
        Predict the target values for a DataFrame.

        Args:
            df: pandas DataFrame, the input data.

        Returns:
            A pandas Series with the predicted target values.
        '''
        if not self.ready:
            raise RuntimeError("Cannot use predict() on an untrained model. Call fit() first.")
        # Create appropriate bins in input data
        if self.bin_suggestions:
            for attr in self.bin_suggestions:
                bin_labels = [f"({self.bin_suggestions[attr][i]}, {self.bin_suggestions[attr][i+1]}]" for i in range(len(self.bin_suggestions[attr])-1)]
                df[attr] = pd.cut(df[attr], bins=self.bin_suggestions[attr], labels=bin_labels)
        return df.apply(lambda record: self.__predict_record(record), axis=1)
    
    
    def __getstate__(self):
        '''
        Prepare the object's state for pickling by excluding non-pickleable attributes.

        This method is automatically called when pickling the object. It creates a copy
        of the object's dictionary (`__dict__`) and excludes specific attributes that
        cannot or should not be pickled. For example, the 'plot' attribute is excluded
        as it contains a matplotlib plot object.

        Returns:
            dict: A dictionary representing the pickleable state of the object.
        '''
        state = self.__dict__.copy()
        state['plot'] = None  # Exclude plot from serialization
        return state


    def __setstate__(self, state):
        '''
        Restore the object's state after unpickling.

        This method is automatically called when unpickling the object. It updates the
        object's dictionary with the saved state and reinitializes specific attributes
        if necessary. For example, it recreates the 'plot' attribute if the model is ready.

        Args:
            state (dict): The dictionary representing the object's state.
        '''
        self.__dict__.update(state)
        if self.ready:  # Recreate the plot if the model is trained and ready
            self.plot = self.__plot_network()
            
    
    def save(self, file_path: str):
        '''
        Save the IFN model to a file using pickle.

        Args:
            file_path (str): Path to the file where the model will be saved.
        
        Raises:
            IOError: If the file cannot be written to.
        '''
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"Model saved to {file_path}.")


    @staticmethod
    def load(file_path: str) -> 'IFN':
        '''
        Load an IFN model from a pickle file.

        Args:
            file_path (str): Path to the file from which the model will be loaded.

        Returns:
            IFN: The loaded IFN model instance.

        Raises:
            IOError: If the file cannot be read.
        '''
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {file_path}.")
        return model