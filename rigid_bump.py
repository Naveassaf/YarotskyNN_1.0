import tensorflow as tf
import numpy as np
from bump import Bump
import Additional_Sources.UndyingReLU as UDrelu

# CONSTANTS
from x_squared import USE_UNDYING_RELU
from bump import BUMP_FUNCTION_WIDTH
BUMP_CENTER_MIN = -0.1
BUMP_CENTER_MAX = 1.1


class RigidBump(Bump):
    def __init__(self, N, bump_index, naming_postfix, input_placeholder=None, graph=None, ud_relus=USE_UNDYING_RELU):
        # Create placeholder dictionary in order to update values of second weights vector and bias vector during training
        self.feed_dict = {}

        Bump.__init__(self, N, bump_index, naming_postfix, input_placeholder=input_placeholder, graph=graph,
                      trainable=True, ud_relus=ud_relus)

    def build_bump_graph(self):

        with self.tf_graph.as_default():
            self.first_weights = tf.constant(value=np.array([1.0, 1.0, 1.0, 1.0]), shape=[1, BUMP_FUNCTION_WIDTH], dtype=tf.float64,
                                             name=self.naming_postfix+'_weights_1')

            first_matmul = tf.matmul(self.input_placeholder, self.first_weights, name=self.naming_postfix+ '_matmul_1')

            # Calculate initial offsets
            x_init = 2 * self.relu_offset
            y_init = self.relu_offset

            # Create three variables, ensuring only three degrees of freedom
            large_OS = tf.math.abs(tf.get_variable(name=self.naming_postfix+'_x_var', dtype=tf.float64, initializer=np.array([x_init])))

            small_OS = tf.math.abs(tf.get_variable(name=self.naming_postfix + '_y_var', dtype=tf.float64, initializer=np.array([y_init])))

            # Ensure bump center ramains between BUMP_CENTER_MIN and BUMP_CENTER_MAX
            temp_center = tf.math.maximum(tf.get_variable(name=self.naming_postfix + '_center_var', dtype=tf.float64, initializer=np.array([self.bump_center])),
                                          tf.constant(value=np.array([BUMP_CENTER_MIN]), shape=[1, 1], dtype=tf.float64))
            self.center_var = tf.math.minimum(temp_center, tf.constant(value=np.array([BUMP_CENTER_MAX]), shape=temp_center.shape, dtype=tf.float64))

            first_biases = tf.concat(values=[large_OS-self.center_var, small_OS-self.center_var,
                                             -small_OS-self.center_var, -large_OS-self.center_var], axis=1)

            adder = tf.add(first_matmul, first_biases, name=self.naming_postfix+'_adder' )

            # If learning net, append batch normalization layer before ReLus
            if self.use_ud_relus:
                relus = UDrelu.relu(input_tnsr=adder, name=self.naming_postfix + '_relu')
            else:
                relus = tf.nn.relu(adder, name=self.naming_postfix + '_relu')

            slope = tf.math.divide(tf.constant(value=np.array([1.0]), shape=[1,1], dtype=tf.float64),large_OS-small_OS)

            second_weights = tf.concat(values=[slope, -slope, -slope, slope], axis=0)

            self.final_output = tf.matmul(relus, second_weights, name=self.naming_postfix+ '_matmul_2')

    def update_bump_center(self, open_session):
        '''
        Given a currently running TD session, calculates the bump's updated center and the corresponding variable
        :param open_session:
        :return:
        '''
        self.bump_center = open_session.run(self.center_var)