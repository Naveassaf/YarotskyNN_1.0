import tensorflow as tf
import numpy as np
import Additional_Sources.UndyingReLU as UDrelu
import math
import matplotlib.pyplot as plt

#CONSTANTS
from x_squared import USE_UNDYING_RELU
DEFAULT_SAMPLING_RESOLUTION = 1e-7
DEFAULT_LEARNING_RATE = 1e-3
MANUAL_LEARNING_RATE_DECAY = True
TRAINING_VS_TEST_RATIO = 7
PRINT_TO_CONSOLE = True
ITERATIONS_PER_PRINT = 1e3
UDRELU_RESURRECTION_FREQ = 1e3
DEFAULT_BATCH_SIZE = 500
MANUAL_LEARNING_RATE_DECAY_TEST = 10
MANUAL_LEARNING_RATE_DECAY_COEF = 0.5
LOSS_PRINT_ITERATION_RANGE = 50
DEFAULT_MSE_RESOLUTION = 1e-3
ADAPTIVE_TAYLOR_RECALC_FREQ = 1000
VIDEO_RESOLUTION = 100
BUMP_PLOT_THRESH = 0.02
TAYLOR_PLOT_THRESH = 0.4

class NetMaster():
    def __init__(self, tf_graph,function, net_output, input_placeholder, output_placeholder,
                 sampling_resolution=DEFAULT_SAMPLING_RESOLUTION, decay_learning_rate=MANUAL_LEARNING_RATE_DECAY,
                 learning_rate=DEFAULT_LEARNING_RATE, using_ud_relu=USE_UNDYING_RELU, batch_size = DEFAULT_BATCH_SIZE,
                 trainable_net=True, adaptive_mode=False, bump_list=[], poly_list=[]):

        # NN's Graph to which learning rate placeholder (for learning rate decay) and other configs shall be added
        self.tf_graph = tf_graph

        # Distance between two adjacent samples in the data (train/validation/test) sets
        self.sampling_resolution = sampling_resolution

        # Function net is trained to implement
        self.function = function

        # Net's output. Used to compute loss during training
        self.final_output = net_output

        # Net's input placeholder
        self.input_placeholder = input_placeholder

        # Nets' output placeholder - also for loss calculation
        self.output_placeholder = output_placeholder

        # Boolean which determines whether learning rate will be controlled by Adam optimizer or manually through PHold.
        self.decay_learning_rate = decay_learning_rate

        # Initial learning rate value whether manually decaying or not
        self.learning_rate = learning_rate

        # Boolean which configures whether the NN's Relus are Undying Relus (imported module) or normal Relus
        self.using_ud_relu = using_ud_relu

        # Dictionary which holds the data sets that will
        self.data_sets = {}

        self.batch_size = batch_size

        self.loss_list = []

        self.trainable_net=trainable_net

        self.adaptive_mode = adaptive_mode

        self.bump_list = bump_list

        self.poly_list = poly_list

        self.feed_dict = {}

        with self.tf_graph.as_default():
            # Create placeholder used for decaying learning rate (only for manual learning decay)
            self.learning_rate_placeholder = tf.placeholder(tf.float64, name="learning_rate")

            # Determine loss function as MSE between label and output of net
            self.loss = tf.losses.mean_squared_error(self.output_placeholder, self.final_output)

            # Calc cost
            self.cost = tf.reduce_mean(tf.cast(self.loss, tf.float64))

            # Use Adam (for now - may later be changed) and set initial learning rate
            if self.decay_learning_rate:
                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder, name='Adam_op')
            else:
                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='Adam_op')

            # Variable used to ensure batch normalization occurs every batch
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            if self.trainable_net:
                with tf.control_dependencies(self.update_ops):
                    self.optimizer = trainer.minimize(self.cost)

            # Initialize all variables before beginning session
            self.init = tf.global_variables_initializer()

        # Declare self.sess as the default (and here only) session, passed as defalut to all session-based functions.
        self.sess = tf.Session(graph=self.tf_graph)

        # Run session initialization
        self.sess.run(self.init)

    def evaluate(self, x, node=None):
        '''
        Evaluates the given network value at points x.

        :param x: Point at which we wish to evaulate NeuralNetwork
        :param node: the node where the net will be evaluated.
        :return: output of net for given input vector
        '''
        # Prepare input place holder before evaluation
        # Check if Taylor coefficients need to be placed into feed_dictionary (for running sessions before training)
        self.feed_dict.update(self.get_feed_dict(x, update_taylor_coeffs=False))
        if self.adaptive_mode and (not self.poly_list[0].coeffs_PH in self.feed_dict):
            self.feed_dict.update(self.get_feed_dict(x, update_taylor_coeffs=True))

        # The wanted value is the output of the sum of the hat functions or the node specified.
        if node == None:
            # If no node is provided, evaluate final_output of net
            output = self.sess.run(self.final_output, feed_dict=self.feed_dict)
        else:
            # Else evaluate node given
            output = self.sess.run(node, feed_dict=self.feed_dict)

        # Output is returned as [1,1] array. Returns its only value
        return output

    def train(self, print_to_console=PRINT_TO_CONSOLE, print_frequecy=ITERATIONS_PER_PRINT, video_directory=None,
              bump_dict=None, taylor_dict=None):

        # Prepare sets for training (created only when training to save time for hardcoded-net evaluation)
        self.data_sets = self.prep_sets()

        # Shuffle training set before using it
        self.shuffle_set('train_set')

        # Perform training batches
        samples_since_resurrection = 0

        # Evenly spaced samples for video
        video_inputs = np.arange(0, 1.001, 0.001)

        for iteration in range(self.iteration_count):
            # Get vectors of (x,y) tuples with for the current iteration
            train_x, train_y = self.next_batch('train_set', iteration * self.batch_size, (iteration + 1) * self.batch_size)

            # Set value of input vector placeholders and output result placeholder.
            # If addaptive_mode, once every ADAPTIVE..FREQ iterations, update bump centers and polynomial coefs accord.
            if iteration % ADAPTIVE_TAYLOR_RECALC_FREQ == 0:
                self.feed_dict.update(self.get_feed_dict(train_x, update_taylor_coeffs=True))
            else:
                self.feed_dict.update(self.get_feed_dict(train_x, update_taylor_coeffs=False))

            # Maintenance of UD Relus in case neural net is using them
            if self.using_ud_relu:
                self.feed_dict = UDrelu.add_to_feed_dict(self.feed_dict)

                # Resurrect Relus every UDRELU_RESURRECTION_FREQ samples
                if samples_since_resurrection >= UDRELU_RESURRECTION_FREQ:
                    UDrelu.resurrect(self.sess, self.feed_dict)
                    samples_since_resurrection = 0
                else:
                    samples_since_resurrection += self.batch_size

            if (iteration % VIDEO_RESOLUTION == 0) and (video_directory != None):
                # Save image to video file
                plt.clf()
                print('Saving photo: {}'.format(iteration / VIDEO_RESOLUTION))

                # Plot updated bumps
                for index in bump_dict.keys():
                    bump_outputs = np.array(self.evaluate(video_inputs, bump_dict[index].final_output))
                    non_zero_indices = np.where(bump_outputs > BUMP_PLOT_THRESH)[0]
                    plt.plot(video_inputs[non_zero_indices], bump_outputs[non_zero_indices], color='m', linewidth=1.0)
                    if taylor_dict != None:
                        taylor_indices = np.where(bump_outputs > TAYLOR_PLOT_THRESH)[0]
                        plt.plot(video_inputs[taylor_indices],
                                 eval_sympy_polynomial_vector(video_inputs[taylor_indices], taylor_dict[index]),
                                 color='r', linewidth=1.0)




                plt.plot(video_inputs, self.function(video_inputs), color='b')
                plt.plot(video_inputs, self.evaluate(video_inputs), color='c')
                plt.savefig(video_directory + '\\{}.png'.format(int(iteration / VIDEO_RESOLUTION)))


            # Calculate and display loss
            try:
                _, batch_loss, x = self.sess.run([self.optimizer, self.loss, self.update_ops], feed_dict=self.feed_dict)
            except:
                print('Training Error --- Recalculating Taylor coefficients and trying again')
                self.feed_dict.update(self.get_feed_dict(train_x, update_taylor_coeffs=True))
                _, batch_loss, x = self.sess.run([self.optimizer, self.loss, self.update_ops], feed_dict=self.feed_dict)

            self.loss_list.append(batch_loss)

            # Decay learning rate if relevant.e
            #TODO change to batch-based decay, and not loss dependent decay
            if (self.decay_learning_rate) and (self.learning_rate / batch_loss > MANUAL_LEARNING_RATE_DECAY_TEST):
                if print_to_console:
                    print('Decaying Learning Rate')
                self.learning_rate *= MANUAL_LEARNING_RATE_DECAY_COEF

            # Print batch number and loss in current batch (condition ensures first loss is printed for reference)
            if print_to_console and (((iteration + 1) % print_frequecy == 0) or (iteration == self.iteration_count - 1) or (iteration == 0)):
                print('Trained iteration: {}/{}, Loss: {}'.format(iteration + 1, self.iteration_count, min(self.loss_list[-LOSS_PRINT_ITERATION_RANGE:])))
                #TODO FOR NOW

    def prep_sets(self):
        '''
        Function used to generate training, validation, and test sets for f(x) = x^2 (for now).

        :return: data: dictionary with training, validation, and test sets
        '''

        data = {}
        x_coordinates = np.arange(0, 1, self.sampling_resolution)
        y_coordinates = x_coordinates ** 2
        sample_count = len(x_coordinates)

        # Shuffle x_coordinates before placing into sets
        permutation = np.random.permutation(sample_count)
        x_coordinates = x_coordinates[permutation]
        y_coordinates = y_coordinates[permutation]

        # Divide the shuffled sample set into validation, training, and test sets
        data['valid_set'] = x_coordinates[0:round(sample_count / TRAINING_VS_TEST_RATIO)]
        data['valid_label'] = y_coordinates[0:round(sample_count / TRAINING_VS_TEST_RATIO)]
        data['test_set'] = x_coordinates[round(sample_count / TRAINING_VS_TEST_RATIO):round(
            2 * sample_count / TRAINING_VS_TEST_RATIO)]
        data['test_label'] = y_coordinates[round(sample_count / TRAINING_VS_TEST_RATIO):round(
            2 * sample_count / TRAINING_VS_TEST_RATIO)]
        data['train_set'] = x_coordinates[round(2 * sample_count / TRAINING_VS_TEST_RATIO):]
        data['train_label'] = y_coordinates[round(2 * sample_count / TRAINING_VS_TEST_RATIO):]

        # Calculate number of iterations necessary to complete training
        self.iteration_count = math.ceil(len(data['train_set']) / self.batch_size)

        return data

    def shuffle_set(self, type):
        '''
        Function called in order to shuffle a data set before training, testing or validation

        :param type: validation, test or training set and corresponding labels to be shuffled
        :return:
        '''
        # Decide on (set, label) pair to be shuffled
        if (type.startswith('train')):
            set = 'train_set'
            label = 'train_label'
        elif (type.startswith('test')):
            set = 'test_set'
            label = 'test_label'
        elif (type.startswith('valid')):
            set = 'valid_set'
            label = 'valid_label'
        else:
            return

        # Once type of set is understood, create random permutation for reordering of set
        permutation = np.random.permutation(len(self.data_sets[set]))
        self.data_sets[set] = self.data_sets[set][permutation]
        self.data_sets[label] = self.data_sets[label][permutation]

    def next_batch(self, type, start, end):
        '''
        Function used to easily get next batch of data during batch learning process

        :param type: 'validation','test' or 'training' parameter
        :param start: starting index of batch sample
        :param end: index one after last sample (x, f(x)) point) in batch
        :return: x_coordinate and corresponding f(x) values arrays for desired batch
        '''

        # Decide on (set, label) pair to be shuffled
        if (type.startswith('train')):
            set = 'train_set'
            label = 'train_label'
        elif (type.startswith('test')):
            set = 'test_set'
            label = 'test_label'
        elif (type.startswith('valid')):
            set = 'valid_set'
            label = 'valid_label'

        x_batch = self.data_sets[set][start:end]
        y_batch = self.data_sets[label][start:end]
        return x_batch, y_batch

    def get_feed_dict(self, x, update_taylor_coeffs=False):
        '''
        Inserts X into input place holder (for calculations) and f(x) into output placeholder (for comparison)

        :param x: list or array-like type. Inputs to the network (list  or array representing the current batch).
        :param f_x: list or array-like type. Outputs of the network (list  or array representing the current batch).
        :return:
        '''
        # Initialize current feed dictionary, inputs and outputs
        inputs = x
        outputs = self.function(x)

        # For manual learning rate decay, sets value of learning rate at beginning of each iteration
        self.feed_dict.update({self.learning_rate_placeholder: self.learning_rate})

        # Create feed dictionary with the shape [#num of inputs, 1]
        self.feed_dict.update({self.input_placeholder: np.array(inputs, dtype='float64').reshape((len(inputs), 1))})
        self.feed_dict.update({self.output_placeholder: np.array(outputs, dtype='float64').reshape((len(outputs), 1))})

        # If in adaptive mode, recalculate taylor coefficients of polynomials necessary PH values to feed dict
        if self.adaptive_mode and update_taylor_coeffs:
            for bump, poly in zip(self.bump_list, self.poly_list):
                bump.update_bump_center(open_session=self.sess)
                self.feed_dict.update(poly.get_updated_feed_dict(taylor_x0=bump.bump_center))


        return self.feed_dict

    def calc_mse(self, function, start=0, stop=1):
        temp_range = np.arange(start, stop+DEFAULT_MSE_RESOLUTION, DEFAULT_MSE_RESOLUTION)
        test_sampling_points = temp_range.reshape([len(temp_range), 1])

        # If placeholder of first polynomial is not in feed dict, none are and thus they must be calced and entered
        self.feed_dict.update(self.get_feed_dict(test_sampling_points, update_taylor_coeffs=False))
        if self.adaptive_mode and (not self.poly_list[0].coeffs_PH in self.feed_dict):
            self.feed_dict.update(self.get_feed_dict(test_sampling_points, update_taylor_coeffs=True))

        [test_outputs] = self.sess.run([self.final_output], feed_dict=self.feed_dict)
        mse = np.average((test_outputs -function(test_sampling_points))**2)

        return mse


def eval_sympy_polynomial_vector(input_array, sympy_polynomial):
        poly_outputs = []
        for x in input_array:
            poly_outputs.append(sympy_polynomial.eval(x))
        return poly_outputs



