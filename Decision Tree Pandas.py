from __future__ import print_function
import csv
import pandas as pd
import os
import matplotlib.pyplot as plt

# csv file name
training_filename = "zoo.csv"
testing_filename = "drift_fish&reptiles_milk&feathers.csv"

# load csv files
training_data = pd.read_csv(training_filename)
testing_data = pd.read_csv(testing_filename)

# create node hit trackers
classes = ["Mammal", "Bird", "Reptile", "Fish", "Amphibian", "Bug", "Invertebrate"]
hits = 0
node_hits = {}
node_misses = {}
data_through_node = {}
node_counter = -1

# create headers dataframe
header = training_data.columns

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = rows["class_type"].value_counts()
    return counts

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

class Question:
    """A Question is used to partition a dataset.
    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows = pd.DataFrame() 
    false_rows = pd.DataFrame()
    for i in range(len(rows)):
        if question.match(rows.iloc[i]):
            true_rows = true_rows.append(rows.iloc[i])
        else:
            false_rows = false_rows.append(rows.iloc[i])
    return true_rows, false_rows


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for i in counts.keys():
        prob_of_lbl = counts[i] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows.iloc[0]) - 1  # number of columns
    for col in range(1, n_features):  # for each feature

        values = set([rows.iloc[i][col] for i in range(len(rows))])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows, number):
        self.predictions = class_counts(rows)
        self.number = number
        node_hits.update({self.number:0})
        node_misses.update({self.number:0})
        data_through_node.update({self.number:pd.DataFrame()})


class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the question, to the parent node, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch, number):
        self.question = question 
        self.number = number
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    """Builds the tree.
    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """
    # Initialize node number
    global node_counter
    number = node_counter
    node_counter += 1

    # Try partitioning the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)
    
    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows, number)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # depending on the answer.c

    return Decision_Node(question, true_branch, false_branch, number)

def graph_tree(node):
    """plot the decision tree model"""
    width_dist = 40
    depth_dist = 40
    # 7 levels since there are 7 classes and 17 attributes
    levels = 8

    fig, ax = plt.subplots()
    ax.set_ylim(-levels * depth_dist, 10)
    ax.set_xlim(-2 * width_dist, 2 * width_dist)

    font = {'family': 'Arial',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
            }
    ax.text(-width_dist / 3, -depth_dist / 2, "T", fontdict=font)
    ax.text(width_dist / 3, -depth_dist / 2, "F", fontdict=font)

    for i in range(1, 8):
        ax.text(-2 * width_dist + 10, -i * 10, str(i) + ": " + classes[i - 1], fontdict=font)

    def graph_recurse(node, x, y, width):
        if isinstance(node, Leaf):
            ax.scatter(x, y)
            count = {}
            for i in node.predictions.keys():
                count[i] = node.predictions[i]
            ax.text(x - 5, y - 10, str(count), fontdict=font)
            return
        ax.text(x + 3, y - 2, str(node.question), fontdict=font)
        if node.true_branch is not None:
            if width < width_dist / 1.5:
                xl = x - width
            else:
                xl = x - width / 2
            yl = y - depth_dist
            ax.scatter(x, y)
            ax.annotate("",
                        xy=(xl, yl), xycoords='data',
                        xytext=(x, y), textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="arc3"),
                        )
            graph_recurse(node.true_branch, xl, yl, width * 0.8)

        if node.false_branch is not None:
            if width < width_dist / 1.5:
                xr = x + width
            else:
                xr = x + width / 2
            yr = y - depth_dist
            ax.scatter(x, y)
            ax.annotate("",
                        xy=(xr, yr), xycoords='data',
                        xytext=(x, y), textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="arc3"),
                        )
            graph_recurse(node.false_branch, xr, yr, width * 0.8)
        return

    graph_recurse(node, 0, 0, width_dist)

    plt.axis('off')
    plt.savefig('Visualization.png')
    os.system("Visualization.png")

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):

        count = {}
        for i in node.predictions.keys():
            count[i] = node.predictions[i]
        print(spacing + 'Predict', count)
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

# Function that retrains the tree by finding the least accurate 
def retrain():
    global training_data
    for key in node_hits:
        if node_hits[key] == 0 or node_misses[key] / node_hits[key] > 0.4:
            training_data = training_data.append(data_through_node[key])
    new_tree = build_tree(training_data)
    print_tree(new_tree)
    graph_tree(new_tree)
    hits = 0
    for i in range(len(testing_data)):
        classifier = classify(testing_data.iloc[i], new_tree)
        print("Actual: %s. Predicted: %s" %
              (testing_data.iloc[i][-1], print_leaf(classifier.predictions)))
        if classifier.predictions.keys()[0]  == testing_data.iloc[i][-1]:
            hits += 1
    accuracy = hits / len(testing_data)
    print("Accuracy: ", accuracy)

if __name__ == '__main__':
    my_tree = build_tree(training_data)
    print_tree(my_tree)
    graph_tree(my_tree)
    # Evaluate
    for i in range(len(testing_data)):
        classifier = classify(testing_data.iloc[i], my_tree)
        print("Actual: %s. Predicted: %s" %
              (testing_data.iloc[i][-1], print_leaf(classifier.predictions)))
        data_through_node[classifier.number] = data_through_node[classifier.number].append(testing_data.iloc[i])
        if classifier.predictions.keys()[0]  == testing_data.iloc[i][-1]:
            node_hits[classifier.number] += 1
            hits += 1
        else:
            node_misses[classifier.number] += 1
    accuracy = hits / len(testing_data)
    print("Accuracy: ", accuracy)
    if accuracy < 0.85:
        print("Drift detected: Model adapting.")
        retrain()