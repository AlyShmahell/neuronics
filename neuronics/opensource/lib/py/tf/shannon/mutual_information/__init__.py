from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf

mutual_information_op = tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'mutual_information_op.so'))
mutual_information = mutual_information_op.mutual_information