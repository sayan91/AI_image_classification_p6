import numpy as np
import data_classification_utils
from util import raiseNotDefined
import random

class Perceptron(object):
    def __init__(self, categories, numFeatures):
        """categories: list of strings
           numFeatures: int"""
        self.categories = categories
        self.numFeatures = numFeatures
        self.weightVectors = {}
        for category in self.categories:
            self.weightVectors[category] = np.zeros(numFeatures)

        """YOUR CODE HERE"""
        # raiseNotDefined()


    def classify(self, sample):
        """sample: np.array of shape (1, numFeatures)
           returns: category with maximum score, must be from self.categories"""

        """YOUR CODE HERE"""
        scores = {}

        for category in self.categories:
            scores[np.dot(self.weightVectors[category], sample)] = category

        return scores[max(scores.keys())]

        # raiseNotDefined()


    def train(self, samples, labels):
        """samples: np.array of shape (numSamples, numFeatures)
           labels: list of numSamples strings, all of which must exist in self.categories
           performs the weight updating process for perceptrons by iterating over each sample once."""

        """YOUR CODE HERE"""

        correct = 0.0
        prevCorrect = len(samples)
        while abs(correct - prevCorrect) > 0.1:
            prevCorrect = correct
            correct = 0.0
            for sample_id in range(len(samples)):
                label = labels[sample_id]
                result = self.classify(samples[sample_id])
                updatedWeight = False
                if result != label:
                    self.weightVectors[result] = self.weightVectors[result] - samples[sample_id]
                    self.weightVectors[label] = self.weightVectors[label] + samples[sample_id]
                else:
                    correct += 1.0
