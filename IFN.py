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
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
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


    def __init__(self, train_data: pd.DataFrame, target: str, P_VALUE_THRESH: float = 0.1, max_depth: int = None, weights_type: str = 'probability'):
        self.train_data = train_data
        self.target = target
        self.P_VALUE_THRESH = P_VALUE_THRESH
        self.input_vars = {col: 0 for col in train_data.columns if col != target}  # Input vars with 1 for 'used'
        self.N = len(train_data)
        self.bin_suggestions = self.__init_create_bin_suggestions()  # A simplified version of threshold optimization
        self.root = IFN.Node(train_data, level_attr='root', color='purple')
        self.target_nodes = self.__init_target_nodes()
        self.nodes = [self.root] + self.target_nodes
        self.levels = []
        self.edges = []  # A tuple like (pointer, poitee, {weight, probability}), all non-terminal nodes get weight == None
        self.max_depth = max_depth
        self.weights_type = weights_type  # 'mutual' or 'probability'
        self.plot = None


    def __init_target_nodes(self):
        '''
        Initialize the target nodes based on the unique values in the target variable.
        '''
        target_values = self.train_data[self.target].unique()
        target_nodes = [IFN.Node(self.train_data[self.train_data[self.target] == val], level_attr=self.target, level_attr_val=val, is_terminal=True, color='red') for val in target_values]
        for target_node in target_nodes:
            logging.info(f'Target Class Initiated: "{target_node.level_attr_val}", No. Records = {len(target_node.data)}')
        return target_nodes


    @time_method_call
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
    def fit(self):
        '''
        Fit the IFN model to the training data.
        '''
        self.__build_network([self.root])
        # Clean unused numerical column names
        self.bin_suggestions = {column: self.bin_suggestions[column] for column in self.bin_suggestions if self.input_vars[column] == 1}
        self.plot = self.__plot_network()


    # def __plot_network(self):
    #     '''
    #     Plot the IFN network.

    #     Returns:
    #         matplotlib.pyplot: The plot object.
    #     '''
    #     G = nx.DiGraph()
    #     colors = []
    #     pos = {}
    #     labels = {}

    #     # Add nodes to the graph
    #     for level_index, level in enumerate(self.levels):
    #         for node_index, node in enumerate(level):
    #             node_id = id(node)
    #             G.add_node(node_id)
    #             colors.append(node.color)
    #             # Assign positions based on level and order within the level
    #             pos[node_id] = (node_index, -level_index)
    #             # Add label if the node is not a root node (purple)
    #             if node.color != 'purple':
    #                 labels[node_id] = node.level_attr_val

    #     # Add edges to the graph, only include weights that are not None
    #     edge_labels = {}
    #     for (parent, child, weight) in self.edges:
    #         parent_id = id(parent)
    #         child_id = id(child)
    #         # Ensure weight is a valid number and not None
    #         if weight is not None:
    #             # Decide which edge label to plot
    #             if self.weights_type == 'probability':
    #                 rounded_weight = round(weight['probability'], 2)
    #             else:
    #                 rounded_weight = round(weight['weight'], 2)
    #             G.add_edge(parent_id, child_id, weight=rounded_weight)
    #             edge_labels[(parent_id, child_id)] = rounded_weight
    #         else:
    #             G.add_edge(parent_id, child_id)

    #     # Create a wider figure
    #     plt.figure(figsize=(13, 5))  # Adjust the width and height as needed
    #     # Draw the graph with bigger nodes and brighter colors
    #     nx.draw(G, pos, with_labels=False, node_color=colors, node_size=1000, alpha=0.6)
    #     nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color='black', font_family='sans-serif', font_weight='bold')

    #     # Draw edge labels with smaller font size and rounded values
    #     edge_label_options = {
    #         "font_size": 7,
    #         "rotate": False,
    #     }
    #     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, **edge_label_options)

    #     # Find the maximum x-coordinate of the nodes
    #     max_x = max(x for x, y in pos.values())

    #     # Draw level names with dashed lines
    #     for level_index, level in enumerate(self.levels):
    #         if level:
    #             first_node = level[0]
    #             first_node_id = id(first_node)
    #             first_node_pos = pos[first_node_id]
    #             level_name = first_node.level_attr  # Assuming all nodes in the level have the same level_attr

    #             # Draw dashed line for the level
    #             plt.plot([0, max_x + 0.1], [first_node_pos[1], first_node_pos[1]], color='gray', linestyle='--', linewidth=0.5)
    #             # Add level name text slightly to the right of the most right node
    #             plt.text(max_x + 0.15, first_node_pos[1], level_name, verticalalignment='center', fontsize=12, color='gray')
    #     return plt
    
    def __plot_network(self):
        '''
        Plot the IFN network.

        Returns:
            matplotlib.pyplot: The plot object.
        '''
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
                # Control node color luminance
                colors.append(node.color if node.color != 'purple' else 'mediumpurple')
                # Assign positions based on level and order within the level
                pos[node_id] = (node_index, -level_index)
                # Add label if the node is not a root node (purple)
                if node.color != 'purple':
                    labels[node_id] = node.level_attr_val
                # Determine terminal nodes
                if not any(child for (parent, child, _) in self.edges if id(parent) == node_id):
                    terminal_nodes.add(node_id)

        # Add edges to the graph, only include weights that are not None
        for (parent, child, weight) in self.edges:
            parent_id = id(parent)
            child_id = id(child)
            # Ensure weight is a valid number and not None
            if weight is not None:
                if self.weights_type == 'probability':
                    edge_weights.append(weight['probability'])
                else:
                    edge_weights.append(weight['weight'])
                G.add_edge(parent_id, child_id)
            else:
                G.add_edge(parent_id, child_id)
                edge_weights.append(0.0)  # Assign a default weight if none is provided

        # Normalize edge weights for colormap
        norm = mcolors.Normalize(vmin=-0.5, vmax=max(edge_weights))
        cmap = plt.cm.get_cmap('Purples')

        # Set edge colors
        edge_colors = []
        for (parent, child) in G.edges():
            if child in terminal_nodes:
                weight = edge_weights.pop(0)
                edge_colors.append(cmap(norm(weight)))
            else:
                edge_colors.append(cmap(0.4))  # Medium grey for non-terminal edges

        # Create a wider figure
        plt.figure(figsize=(13, 5))  # Adjust the width and height as needed

        # Draw the graph with bigger nodes and brighter colors
        nx.draw(G, pos, with_labels=False, node_color=colors, node_size=800, alpha=0.6, edge_color=edge_colors)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=5, font_color='black', font_family='sans-serif', font_weight='bold')

        # Add a color bar as a legend
        norm = mcolors.Normalize(vmin=0, vmax=max(edge_weights))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Edge Weight')
    
        # Find the maximum x-coordinate of the nodes
        max_x = max(x for x, y in pos.values())

        # Draw level names with dashed lines
        for level_index, level in enumerate(self.levels):
            if level:
                first_node = level[0]
                first_node_id = id(first_node)
                first_node_pos = pos[first_node_id]
                level_name = first_node.level_attr  # Assuming all nodes in the level have the same level_attr

                # Draw dashed line for the level
                plt.plot([0, max_x + 0.1], [first_node_pos[1], first_node_pos[1]], color='gray', linestyle='--', linewidth=0.5)
                # Add level name text slightly to the right of the most right node
                plt.text(max_x + 0.15, first_node_pos[1], level_name, verticalalignment='center', fontsize=9, color='gray')
         
        return plt

    
    def show(self):
        '''
        Display the plot of the IFN network.
        '''
        self.plot.show()


    def calculate_min_error_probability(self) -> float:
        '''
        Calculate the minimum prediction error P_e based on Fano's Inequality using the complete feature set.

        Returns:
            float, the minimum prediction error P_e.
        '''
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
        # Create appropriate bins in input data
        if self.bin_suggestions:
            for attr in self.bin_suggestions:
                bin_labels = [f"({self.bin_suggestions[attr][i]}, {self.bin_suggestions[attr][i+1]}]" for i in range(len(self.bin_suggestions[attr])-1)]
                df[attr] = pd.cut(df[attr], bins=self.bin_suggestions[attr], labels=bin_labels)
        return df.apply(lambda record: self.__predict_record(record), axis=1)