import tensorflow as tf
import numpy as np
from polynomial import Polynomial


# CONSTANTS
from x_squared import USE_UNDYING_RELU
RANDOM_COEFF_RANGE = (1, 3)

class AdaptivePolynomial(Polynomial):
    def __init__(self, naming_postfix, coefficients=[],input_placeholder=None,
                 graph=None, ud_relus=USE_UNDYING_RELU, trainable=True, polynomial_degree=None):

        self.feed_dict = {}

        if len(coefficients) == 0:
            # Create random coefficients for the polynomial
            coefficients = np.random.random(size=polynomial_degree) * (RANDOM_COEFF_RANGE[1] - RANDOM_COEFF_RANGE[0])\
                           + RANDOM_COEFF_RANGE[0]

        Polynomial.__init__(self, net_degree=None, naming_postfix=naming_postfix, coefficients=coefficients
                            ,input_placeholder=input_placeholder, graph=graph, yarotsky_initialization=False,
                            ud_relus=ud_relus, tf_squaring_modules=False, trainable=trainable)
    def build_net(self):

        with self.tf_graph.as_default():
            # Iterate over necessary monomials, appending each level to the graph based on Yarotsky's architecture
            for monomial_degree in range(self.polynomial_degree+1):

                # Deal with constant (X^0)
                if monomial_degree == 0:
                    self.constant_tensor = tf.constant(value=self.coefficients[0], shape=[1, 1], dtype=tf.float64)
                    self.monomial_outputs.append(self.constant_tensor)

                # Deal with X^1
                elif monomial_degree == 1:
                    self.monomial_outputs.append(self.input_placeholder)

                # Deal with all degrees larger than 1 (x^2, x^3, x^4, ...)
                else:
                    self.monomial_outputs.append(tf.math.pow(x=self.input_placeholder, y=float(monomial_degree)))

            # Now "link" the monomials - multiply them by their corresponding coefficients and sum them up
            self.monomial_tensor = tf.concat(self.monomial_outputs[1:], name=self.naming_postfix+'_monomial_vector', axis=1)

            # Create coefficient tensor without constant (which will be added after)
            initer = tf.constant(value=self.coefficients[1:], shape=[len(self.coefficients) - 1, 1], dtype=tf.float64)

            tf.placeholder(dtype=tf.float64, shape=[None, 1], name='output_placeholder')


            coefficient_tensor = tf.get_variable(name=self.naming_postfix+'_coeffs', dtype=tf.float64, initializer=initer, trainable=self.trainable)

            matmul = tf.matmul(self.monomial_tensor, coefficient_tensor, name=self.naming_postfix+'_output')

            self.final_output = tf.add(matmul, self.constant_tensor)