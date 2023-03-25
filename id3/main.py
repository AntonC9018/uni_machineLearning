import pandas
import numpy

def main():
    data = pandas.read_csv('data.csv')
    label = "joc"
    tree = id3(data, label)

    Printer().print_node(tree)

    test_data = pandas.read_csv('data_test.csv')
    run_decision_tree_and_print_results(tree, test_data)
    
    # classifications = run_decision_tree(tree, data)
    # print(classifications == data[label])

class TreeNode:
    children : "list[TreeNode]" = None
    children_feature : str = None
    feature_value : str = None
    classification : str = None
    data : pandas.DataFrame = None

    @property
    def is_leaf(self):
        return self.children is None

def run_decision_tree_and_print_results(tree: TreeNode, data: pandas.DataFrame):
    classifications = run_decision_tree(tree, data)

    for i in range(0, len(classifications)):
        # print the row and the classification
        print(f"{data.iloc[i]} -> {classifications[i]}")

def run_decision_tree(tree: TreeNode, data: pandas.DataFrame) -> "list[str]":
    results = []
    for index, row in data.iterrows():
        node = tree
        while not node.is_leaf:
            feature = node.children_feature
            value = row[feature]
            for child in node.children:
                if child.feature_value == value:
                    node = child
                    break
        results.append(node.classification)
    return results


def id3(data: pandas.DataFrame, label: str) -> TreeNode:
    if data.shape[0] == 0:
        return None

    root = TreeNode()
    root.data = data
    attribute_values = {column: list(data[column].unique()) for column in data.columns}
    return _id3_internal(root, label, attribute_values)
    
# https://www.wikiwand.com/en/ID3_algorithm#/Pseudocode
def _id3_internal(node: TreeNode, label: str, attribute_values: "dict[str, list[str]]"):
    data = node.data
    
    num_rows = data.shape[0]
    assert(num_rows != 0)

    class_counts = data[label].value_counts()
    most_common_class = class_counts.index[0]
    if num_rows == class_counts[0]:
        node.classification = most_common_class
        return node

    if len(data.columns) == 1:
        node.classification = most_common_class
        return node

    max_info_gain = 0.0
    best_attr = None
    for attr in data.columns:
        if attr == label:
            continue
        information_gain = get_information_gain(attr, data, label)
        if information_gain > max_info_gain:
            max_info_gain = information_gain
            best_attr = attr

    if best_attr is None:
        node.classification = most_common_class
        return node
            
    node.children_feature = best_attr
    node.children = []

    attr_column = data[best_attr]
    for value in attribute_values[best_attr]:
        partition = data[attr_column == value]

        child = TreeNode()
        child.feature = best_attr
        child.feature_value = value
        node.children.append(child)
        
        if (partition.shape[0] == 0):
            child.classification = most_common_class
        else:
            partition_without_attribute = partition.drop(best_attr, axis=1)
            child.data = partition_without_attribute
            _id3_internal(child, label, attribute_values)

    return node

def get_entropy(data: pandas.DataFrame, label: str):
    counts = data[label].value_counts()
    
    num_rows = data.shape[0]
    entropy = 0.0
    for value in counts:
        p = value / num_rows
        entropy += -p * numpy.log2(p)

    return entropy

def get_information_gain(feature: str, data: pandas.DataFrame, label: str):
    feature_values = data[feature].unique()
    num_rows = data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_values:
        partition = data[data[feature] == feature_value]
        entropy = get_entropy(partition, label)
        feature_info += len(partition) / num_rows * entropy
        
    return get_entropy(data, label) - feature_info

class Printer:
    def __init__(self):
        self.id = 0

    def print_node(self, node: TreeNode):
        name = f"A{self.id}";
        if node.is_leaf:
            if node.classification is None:
                print(f"{name}(\"feature = {node.feature_value}\")")
            else:
                print(f"{name}(\"result = {node.classification}\")")
        else:
            print(f"{name}(\"decision: {node.children_feature}\")")

            for child in node.children:
                self.id += 1
                
                print(f"{name} -- {child.feature_value} --> A{self.id}")
                self.print_node(child)

if __name__ == '__main__':
    main();