import tensorflow as tf
import numpy as np
import pandas as pd
import unittest
from neuronics.opensource.lib.py.tf.shannon import mutual_information

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
        return [int("".join(item)) for item in Data.data[:, list(X)].astype(str)]

def py_entropy(X):
    values = Data.get(X)
    total = float(len(values))
    _, frequencies = np.unique(values, return_counts=True)
    probabilities = frequencies/total
    return -np.sum([probabilities[i] * np.log2(probabilities[i]) for i in range(len(probabilities))])

def py_mutual_information(X, Y):
    return py_entropy(X) + py_entropy(Y) - py_entropy(X | Y)

class conditionalEntropyTestCase(unittest.TestCase):
    
    def test_mutual_information(self):
        i1 = tf.placeholder(dtype=tf.int64, shape=(None,None))
        i2 = tf.placeholder(dtype=tf.int64, shape=(None,None))
        centr = mutual_information(i1, i2)
        with tf.Session() as sess:
            tf_result = sess.run(centr, feed_dict={i1: [Data.get(set([0])), Data.get(set([1]))], i2: [Data.get(set([1])), Data.get(set([0]))]}).round(decimals=6)
        py_result = np.array([py_mutual_information(set([0]), set([1])).round(decimals=6), py_mutual_information(set([1]), set([0])).round(decimals=6)])
        print(f"{tf_result}, {py_result}")
        assert f"{tf_result}" == f"{py_result}"

if __name__ == '__main__':
    unittest.main()