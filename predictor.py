import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope
import numpy as np

# Yitao-TLS-Begin
import os
import sys
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python.framework import tensor_util

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS
# Yitao-TLS-End


def resBlock(x, num_outputs, kernel_size=4, stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, scope=None):
    # num_outputs must be divided by channel_factor(2 here)
    assert num_outputs % 2 == 0
    with tf.variable_scope(scope, 'resBlock'):
        shortcut = x
        if stride != 1 or x.get_shape()[3] != num_outputs:
            shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride,
                                  activation_fn=None, normalizer_fn=None, scope='shortcut')
        x = tcl.conv2d(x, num_outputs / 2, kernel_size=1,
                       stride=1, padding='SAME')
        x = tcl.conv2d(x, num_outputs / 2, kernel_size=kernel_size,
                       stride=stride, padding='SAME')
        x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1,
                       activation_fn=None, padding='SAME', normalizer_fn=None)

        x += shortcut
        x = normalizer_fn(x)
        x = activation_fn(x)
    return x


class resfcn256(object):

    def __init__(self, resolution_inp=256, resolution_op=256, channel=3, name='resfcn256'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op

    def __call__(self, x, is_training=True):
        with tf.variable_scope(self.name) as scope:
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu,
                               normalizer_fn=tcl.batch_norm,
                               biases_initializer=None,
                               padding='SAME',
                               weights_regularizer=tcl.l2_regularizer(0.0002)):
                    size = 16
                    # x: s x s x 3
                    se = tcl.conv2d(x, num_outputs=size,
                                    kernel_size=4, stride=1)  # 256 x 256 x 16
                    se = resBlock(se, num_outputs=size * 2,
                                  kernel_size=4, stride=2)  # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 2,
                                  kernel_size=4, stride=1)  # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 4,
                                  kernel_size=4, stride=2)  # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 4,
                                  kernel_size=4, stride=1)  # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 8,
                                  kernel_size=4, stride=2)  # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 8,
                                  kernel_size=4, stride=1)  # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 16,
                                  kernel_size=4, stride=2)  # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 16,
                                  kernel_size=4, stride=1)  # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 32,
                                  kernel_size=4, stride=2)  # 8 x 8 x 512
                    se = resBlock(se, num_outputs=size * 32,
                                  kernel_size=4, stride=1)  # 8 x 8 x 512

                    pd = tcl.conv2d_transpose(
                        se, size * 32, 4, stride=1)  # 8 x 8 x 512
                    pd = tcl.conv2d_transpose(
                        pd, size * 16, 4, stride=2)  # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(
                        pd, size * 16, 4, stride=1)  # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(
                        pd, size * 16, 4, stride=1)  # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(
                        pd, size * 8, 4, stride=2)  # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(
                        pd, size * 8, 4, stride=1)  # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(
                        pd, size * 8, 4, stride=1)  # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(
                        pd, size * 4, 4, stride=2)  # 64 x 64 x 64
                    pd = tcl.conv2d_transpose(
                        pd, size * 4, 4, stride=1)  # 64 x 64 x 64
                    pd = tcl.conv2d_transpose(
                        pd, size * 4, 4, stride=1)  # 64 x 64 x 64

                    pd = tcl.conv2d_transpose(
                        pd, size * 2, 4, stride=2)  # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(
                        pd, size * 2, 4, stride=1)  # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(
                        pd, size, 4, stride=2)  # 256 x 256 x 16
                    pd = tcl.conv2d_transpose(
                        pd, size, 4, stride=1)  # 256 x 256 x 16

                    pd = tcl.conv2d_transpose(
                        pd, 3, 4, stride=1)  # 256 x 256 x 3
                    pd = tcl.conv2d_transpose(
                        pd, 3, 4, stride=1)  # 256 x 256 x 3
                    # , padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
                    pos = tcl.conv2d_transpose(
                        pd, 3, 4, stride=1, activation_fn=tf.nn.sigmoid)

                    return pos

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class PosPrediction():

    def __init__(self, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1

        # network type
        self.network = resfcn256(self.resolution_inp, self.resolution_op)

        # net forward
        self.x = tf.placeholder(
            tf.float32, shape=[None, self.resolution_inp, self.resolution_inp, 3])
        self.x_op = self.network(self.x, is_training=False)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        # self.sess = tf.Session()

        host = 'localhost'
        port = '8500'
        channel = grpc.insecure_channel('0.0.0.0:8500')
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    def restore(self, model_path):
        # Yitao: commented this restore() call out, for debugging...
        # tf.train.Saver(self.network.vars).restore(self.sess, model_path)
        return

    def predict(self, image):
        # pos = self.sess.run(self.x_op,
        #             feed_dict = {self.x: image[np.newaxis, :,:,:]})

        # # Yitao-TLS-Begin
        # export_path_base = "exported_prnet"
        # export_path = os.path.join(
        #     compat.as_bytes(export_path_base),
        #     compat.as_bytes(str(FLAGS.model_version)))
        # print('Exporting trained model to', export_path)
        # builder = saved_model_builder.SavedModelBuilder(export_path)

        # tensor_info_x = tf.saved_model.utils.build_tensor_info(self.x)
        # tensor_info_y = tf.saved_model.utils.build_tensor_info(self.x_op)

        # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        #     inputs={'input': tensor_info_x},
        #     outputs={'output': tensor_info_y},
        #     method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        # builder.add_meta_graph_and_variables(
        #     self.sess, [tf.saved_model.tag_constants.SERVING],
        #     signature_def_map={
        #         'predict_images':
        #             prediction_signature,
        #     },
        #     legacy_init_op=legacy_init_op)

        # builder.save()

        # print('Done exporting!')
        # # Yitao-TLS-End

        new_image = image[np.newaxis, :, :, :]
        new_image = new_image.astype(np.float32)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'exported_prnet'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['input'].CopyFrom(
            tf.contrib.util.make_tensor_proto(new_image, shape=new_image.shape))
        result = self.stub.Predict(request, 10.0)  # 10 secs timeout

        pos = tensor_util.MakeNdarray(result.outputs['output'])

        # print(pos.shape)

        pos = np.squeeze(pos)

        # print(pos.shape)

        return pos * self.MaxPos

    def predict_batch(self, images):
        pos = self.sess.run(self.x_op,
                            feed_dict={self.x: images})
        return pos * self.MaxPos
