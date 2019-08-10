import tensorflow as tf
import numpy as np
import pandas as pd
import unittest
from neuronics.opensource.lib.py.tf.shannon import mutual_information_grad

class Data:
    dataframe = pd.DataFrame(
        {
            0: [1., 6., 5., 4., 9., 10.],
            1: [1., 6., 8., 8., 9., 10.]
        }
    )
    data = np.asarray(dataframe)
    @classmethod
    def get(cls, X):
        return [float("".join(item)) for item in Data.data[:, list(X)].astype(str)]

class conditionalEntropyTestCase(unittest.TestCase):
    
    def test_mutual_information(self):
        i1 = tf.placeholder(dtype=tf.float32, shape=(None,None))
        i2 = tf.placeholder(dtype=tf.float32, shape=(None,None))
        centr = mutual_information_grad(i1, i2, i2)
        with tf.Session() as sess:
            tf_result = sess.run(centr, feed_dict={i1: [Data.get(set([0])), Data.get(set([1]))], i2: [Data.get(set([1])), Data.get(set([0]))]})
        print(f"{tf_result}")

if __name__ == '__main__':
    unittest.main()