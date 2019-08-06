import tensorflow as tf
import numpy as np
import pandas as pd
import unittest
from neuronics.opensource.lib.py.tf.shannon import conditional_entropy

class Data:
    dataframe = pd.DataFrame(
        {
            0: [1, 6, 5, 4, 9, 10],
            1: [1, 6, 8, 8, 9, 10]
        }
    )
    data = np.asarray(dataframe)
    @classmethod
    def get(cls, X):
        return ["".join(item) for item in Data.data[:, list(X)].astype(str)]

def py_entropy(X):
    values = Data.get(X)
    total = float(len(values))
    _, frequencies = np.unique(values, return_counts=True)
    probabilities = frequencies/total
    return -np.sum([probabilities[i] * np.log2(probabilities[i]) for i in range(len(probabilities))])

def py_conditional_entropy(X, Y):
    return py_entropy(X | Y) - py_entropy(Y)

class conditionalEntropyTestCase(unittest.TestCase):
    
    def test_conditional_entropy(self):
        i1 = tf.placeholder(dtype=tf.string, shape=(None,None))
        i2 = tf.placeholder(dtype=tf.string, shape=(None,None))
        centr = conditional_entropy(i1, i2)
        with tf.Session() as sess:
            tf_result = sess.run(centr, feed_dict={i1: [Data.get(set([0])), Data.get(set([1]))], i2: [Data.get(set([1])), Data.get(set([0]))]}).round(decimals=7)
        py_result = np.array([py_conditional_entropy(set([0]), set([1])).round(decimals=7), py_conditional_entropy(set([1]), set([0])).round(decimals=7)])
        assert f"{tf_result}" == f"{py_result}"

if __name__ == '__main__':
    unittest.main()