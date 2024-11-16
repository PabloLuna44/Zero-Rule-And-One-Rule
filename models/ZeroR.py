import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class ZeroR:
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target
        self.most_common_class = None

    def fit(self):
        # Find the most common class in the target
        self.most_common_class = self.dataset[self.target].mode()[0]

    def predict(self, X):
        # Return the most common class for all samples
        return [self.most_common_class] * len(X)

    def evaluate(self, n_iterations=10):
        accuracies = []

        for _ in range(n_iterations):
            # Split the data into training and testing sets
            train, test = train_test_split(self.dataset, test_size=0.3)

            # Create a new instance of ZeroR and fit
            zero_r = ZeroR(train, self.target)
            zero_r.fit()

            # Make predictions on the test set
            test_predictions = zero_r.predict(test)  
            accuracy = accuracy_score(test[self.target], test_predictions)  # Calculate accuracy
            accuracies.append(accuracy)

        # Return the average accuracy
        avg_accuracy = sum(accuracies) / len(accuracies)
        return avg_accuracy

