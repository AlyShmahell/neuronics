from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
from tensorflow.python.framework import ops

entropy_op = tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'entropy/entropy_op.so'))
entropy = entropy_op.entropy

joint_entropy_op = tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'joint_entropy/joint_entropy_op.so'))
joint_entropy = joint_entropy_op.joint_entropy

conditional_entropy_op = tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'conditional_entropy/conditional_entropy_op.so'))
conditional_entropy = conditional_entropy_op.conditional_entropy

mutual_information_op = tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'mutual_information/mutual_information_op.so'))
mutual_information = mutual_information_op.mutual_information

mutual_information_grad = tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'mutual_information_grad/mutual_information_grad.so'))
mutual_information_grad = mutual_information_grad.mutual_information_grad 
@ops.RegisterGradient("MutualInformation")
def _inner_product_grad_cc(op, grad):
    return grad, grad