import tensorflow as tf
import numpy as np
from bump import Bump

# CONSTANTS
from x_squared import USE_UNDYING_RELU
from bump import BUMP_FUNCTION_WIDTH

class PowerBump(Bump):
    def __init__(self, N, bump_index, naming_postfix, input_placeholder=None, graph=None, ud_relus=USE_UNDYING_RELU):
        # Create placeholder dictionary in order to update values of second weights vector and bias vector during training
        self.feed_dict = {}

        Bump.__init__(self, N, bump_index, naming_postfix, input_placeholder=input_placeholder, graph=graph,
                      trainable=True, ud_relus=ud_relus)

    def get_bump_weights(self, first, name):
        '''
        Function used to create weights object for bump function nets.

        :param first: Boolean which indicates if this is the first or second layer's weights within this bump function
        :param name: The name of the current weights TF object being created.
        :return: TF variable object.
        '''
        slope = 1.0 / self.relu_offset

        if first:
            return tf.constant(value=np.array([1, 1, 1, 1]), shape=[1, BUMP_FUNCTION_WIDTH], dtype=tf.float64, name=name)
        else:
            self.weights_PH = tf.placeholder(name=name, dtype=tf.float64, shape=[BUMP_FUNCTION_WIDTH, 1])
            self.feed_dict[self.weights_PH] = np.array([slope, -slope, -slope, slope]).reshape([BUMP_FUNCTION_WIDTH, 1])

            return self.weights_PH

    def get_bump_biases(self, name):
        '''
        Function used to create biases object for bump function net.

        :param first: Boolean which indicates if this is the first or second layer's biases within this bump function NN.
        :param name: The name of the current biases TF object being made.
        :return: var: TF variable object.
        '''

        # Prep variable initializer function (init to 1 or yarotsky biases)
        variable_init = tf.constant(value=np.array([-(self.bump_center - 2 * self.relu_offset), -(self.bump_center - self.relu_offset),
                            -(self.bump_center + self.relu_offset)]), shape=[1, BUMP_FUNCTION_WIDTH-1], dtype=tf.float64, name=name)

        self.bias_PH = tf.placeholder(shape = [1,1], dtype=tf.float64, name=name+'_PH')

        self.feed_dict[self.bias_PH] = np.array([-(self.bump_center + 2 * self.relu_offset)]).reshape([1,1])

        # Create variable
        self.freedom_elements = tf.get_variable(name=name, dtype=tf.float64, initializer=variable_init, trainable=self.trainable)

        return tf.concat([self.freedom_elements, self.bias_PH], axis=1, name=name)

    def get_updated_feed_dict(self, session):
        '''
        Calculate 4th bias element and new weights_2 vector based on session's current bias values (first 3 elements)
        :param session: current active session of net master calling this function
        :return:
        '''
        current_biases = session.run([self.freedom_elements])[0][0]

        # Calc new bump center (as average of two center points)
        self.bump_center = (current_biases[1] + current_biases[2])/2.0

        # Update fourth bias based on first three biases (preserve symmetry)
        first_offset = current_biases[1]-current_biases[0]
        self.feed_dict[self.bias_PH] = np.array([current_biases[2]+first_offset]).reshape([1,1])

        # Update second weights PH to desired slope
        slope = abs(1.0/first_offset)
        self.feed_dict[self.weights_PH] = np.array([slope, -slope, -slope, slope]).reshape([BUMP_FUNCTION_WIDTH, 1])

        return self.feed_dict