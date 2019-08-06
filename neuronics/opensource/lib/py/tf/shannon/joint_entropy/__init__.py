from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf

joint_entropy_op = tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'joint_entropy_op.so'))
joint_entropy = joint_entropy_op.joint_entropy