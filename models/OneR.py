import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class OneR:
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target
        self.best_feature = None
        self.best_rules = {}

    
    def fit(self):
        highest_accuracy = 0

        # Iterates over each attribute in the dataset
        for attribute in self.dataset.columns:
            if attribute == self.target:
                continue

            # Create a set of rules based on the most common class for each attribute value
            attribute_rules = {}
            for attribute_value in self.dataset[attribute].unique():
                most_common_class = self.dataset[self.dataset[attribute] == attribute_value][self.target].mode()[0]
                attribute_rules[attribute_value] = most_common_class

            # Make predictions using the rules
            attribute_predictions = self.dataset[attribute].map(attribute_rules)
            attribute_accuracy = accuracy_score(self.dataset[self.target], attribute_predictions)

            # Update the best feature if the accuracy is higher
            if attribute_accuracy > highest_accuracy:
                highest_accuracy = attribute_accuracy
                self.best_feature = attribute
                self.best_rules = attribute_rules
        
 
    def predict(self, X):
        # Return the most common class for all samples
        return X[self.best_feature].map(self.best_rules)

    def evaluate(self, n_iterations=10):
        accuracies = []

        for _ in range(n_iterations):
            # Split the data into training and testing sets
            train, test = train_test_split(self.dataset, test_size=0.3)

            # create a new instance of OneR and fit
            one_r = OneR(train, self.target)
            one_r.fit()

            # Make predictions on the test set
            test_predictions = one_r.predict(test)

            # Calculate accuracy
            accuracy = accuracy_score(test[self.target], test_predictions)
            accuracies.append(accuracy)

        # Return the average accuracy
        avg_accuracy = sum(accuracies) / len(accuracies)
        return avg_accuracy

