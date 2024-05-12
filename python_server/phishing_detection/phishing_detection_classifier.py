import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from joblib import dump


def main():
    training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)
    inputs = training_data[:, :-1]
    outputs = training_data[:, -1]

    training_inputs = inputs[:2000]
    training_outputs = outputs[:2000]
    testing_inputs = inputs[2000:]
    testing_outputs = outputs[2000:]

    # fit classifier Decision Tree 
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(training_inputs, training_outputs)

    # save model
    dump(classifier, 'decision_tree_model.joblib')

    # test accuracy
    predictions = classifier.predict(testing_inputs)
    accuracy = 100.0 * accuracy_score(testing_outputs, predictions)
    print(f"Model Accuracy  : {accuracy}")


if __name__ == "__main__":
    main()
