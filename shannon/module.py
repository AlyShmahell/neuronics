from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf

shannon = tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'shannon.so'))
entropy = shannon.entropy
conditional_entropy = shannon.conditional_entropy
joint_entropy = shannon.joint_entropy
mutual_information = shannon.mutual_information
