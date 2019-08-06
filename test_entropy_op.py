import tensorflow as tf
import numpy as np
import pandas as pd
import unittest
from neuronics.opensource.lib.py.tf.shannon import entropy

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

class EntropyTestCase(unittest.TestCase):
    
    def test_entropy(self):
        i1 = tf.placeholder(dtype=tf.string, shape=(None, None))
        e = entropy(i1)
        with tf.Session() as sess:
            tf_result = sess.run(e, feed_dict={i1: [Data.get(set([0])), Data.get(set([1]))]}).round(decimals=7)
        py_result = np.array([py_entropy(set([0])).round(decimals=7), py_entropy(set([1])).round(decimals=7)])
        assert f"{tf_result}" == f"{py_result}"

if __name__ == '__main__':
    unittest.main()