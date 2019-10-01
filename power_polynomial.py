import tensorflow as tf
import numpy as np
import sympy as sy
from polynomial import Polynomial


# CONSTANTS
from x_squared import USE_UNDYING_RELU
RANDOM_COEFF_RANGE = (-5, 5)

class PowerPolynomial(Polynomial):
    def __init__(self, naming_postfix, coefficients=[],input_placeholder=None,
                 graph=None, ud_relus=USE_UNDYING_RELU, polynomial_degree=None, sympy_expr=None):

        if len(coefficients) == 0:
            # Create random coefficients for the polynomial
            coefficients = np.random.random(size=polynomial_degree) * (RANDOM_COEFF_RANGE[1] - RANDOM_COEFF_RANGE[0])\
                           + RANDOM_COEFF_RANGE[0]

        self.feed_dict = {}

        Polynomial.__init__(self, net_degree=None, naming_postfix=naming_postfix, coefficients=coefficients
                            ,input_placeholder=input_placeholder, graph=graph, yarotsky_initialization=False,
                            ud_relus=ud_relus, tf_squaring_modules=False, trainable=False)

    def build_net(self):

        with self.tf_graph.as_default():
            # Iterate over necessary monomials, appending each level to the graph based on Yarotsky's architecture
            for monomial_degree in range(self.polynomial_degree+1):

                # Deal with constant (X^0)
                if monomial_degree == 0:
                    self.constant_tensor = tf.placeholder(shape=[1, 1], dtype=tf.float64)
                    self.feed_dict[self.constant_tensor] = np.array([self.coefficients[0]]).reshape([1,1])
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
            coeffs_values = np.array(self.coefficients[1:]).reshape([len(self.coefficients) - 1, 1])
            self.coefficient_PH = tf.placeholder(name=self.naming_postfix+'_coeffs', dtype=tf.float64, shape=[len(self.coefficients) - 1, 1])
            self.feed_dict[self.coefficient_PH] = coeffs_values

            matmul = tf.matmul(self.monomial_tensor, self.coefficient_PH, name=self.naming_postfix+'_output')

            self.final_output = tf.add(matmul, self.constant_tensor)

    def get_updated_feed_dict(self, sess, new_x0):
        return self.feed_dict