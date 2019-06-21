import tensorflow as tf
import numpy as np
import pandas as pd
import shannon

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

def py_joint_entropy(X, Y):
    return py_entropy(X | Y)

def py_mutual_information(X, Y):
    return py_entropy(X) + py_entropy(Y) - py_entropy(X | Y)


class ShowcaseShannon:

    def __init__(self):
        self.test_entropy()
        self.test_conditional_entropy()
        self.test_joint_entropy()
        self.test_mutual_information()

    def test_entropy(self):
        i1 = tf.placeholder(dtype=tf.string, shape=(None, None))
        e = shannon.entropy(i1)
        with tf.Session() as sess:
            tf_result = sess.run(e, feed_dict={i1: [Data.get(set([0])), Data.get(set([1]))]})
        py_result = [py_entropy(set([0])), py_entropy(set([1]))]
        print(f"{tf_result}, {py_result}")

    def test_conditional_entropy(self):
        i1 = tf.placeholder(dtype=tf.int32, shape=(None,None))
        i2 = tf.placeholder(dtype=tf.int32, shape=(None,None))
        ce = shannon.conditional_entropy(i1, i2)
        with tf.Session() as sess:
            tf_result = sess.run(ce, feed_dict={i1: [Data.get(set([0])), Data.get(set([1]))], i2: [Data.get(set([1])), Data.get(set([0]))]})
        py_result = [py_conditional_entropy(set([0]), set([1])), py_conditional_entropy(set([1]), set([0]))]
        print(f"{tf_result}, {py_result}")

    def test_joint_entropy(self):
        i1 = tf.placeholder(dtype=tf.int64, shape=(None,None))
        i2 = tf.placeholder(dtype=tf.int64, shape=(None,None))
        ce = shannon.joint_entropy(i1, i2)
        with tf.Session() as sess:
            tf_result = sess.run(ce, feed_dict={i1: [Data.get(set([0])), Data.get(set([1]))], i2: [Data.get(set([1])), Data.get(set([0]))]})
        py_result = [py_joint_entropy(set([0]), set([1])), py_joint_entropy(set([1]), set([0]))]
        print(f"{tf_result}, {py_result}")

    def test_mutual_information(self):
        i1 = tf.placeholder(dtype=tf.float32, shape=(None,None))
        i2 = tf.placeholder(dtype=tf.float32, shape=(None,None))
        ce = shannon.mutual_information(i1, i2)
        with tf.Session() as sess:
            tf_result = sess.run(ce, feed_dict={i1: [Data.get(set([0])), Data.get(set([1]))], i2: [Data.get(set([1])), Data.get(set([0]))]})
        py_result = [py_mutual_information(set([0]), set([1])), py_mutual_information(set([1]), set([0]))]
        print(f"{tf_result}, {py_result}")

if __name__ == '__main__':
    ShowcaseShannon()