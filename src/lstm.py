import tensorflow as tf
import numpy as np
import os

class lstm_configuration :

    def __init__(self, num_chunks, chunk_size, hidden_units, hidden_layers, samples,output_classes):

        self.num_inputs = chunk_size
        self.batch_size = 512
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.output_classes = 6
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.samples = samples

        self.learning_rate = 0.005
        self.lambda_loss = 0.0015
        self.training_epochs = 300

        with tf.name_scope('weights') :
            self.W = {
                'hidden' : tf.Variable(tf.random_normal([chunk_size, hidden_units]), name = 'input_hidden_weights'),
                'output' : tf.Variable(tf.random_normal([hidden_units, output_classes]), name = 'hidden_output_weights')
            }
        with tf.name_scope('bias') :
            self.bias ={
                'hidden' : tf.Variable( tf.random_normal([hidden_units]), name = 'input_hidden_bias'),
                'output' : tf.Variable( tf.random_normal([output_classes]), name = 'hidden_output_bias')
            }

def lstm ( features, lstm_config ) :

    '''

    :param features: (num_batches, num_chunks, chunk_size) (num_batches, 187, 3)
    :param lstm_config:
    :return:
    '''

    chunk_size = lstm_config.chunk_size
    num_chunks = lstm_config.num_chunks

    layer = {}
    #transposing features to (num_chunks, num_batches, chunk_size)
    fetaures_t = tf.transpose(features, [1,0,2])
    features_reshape = tf.reshape(fetaures_t, [-1, chunk_size])

    print features_reshape.get_shape()
    print lstm_config.W['hidden']

    features_w = tf.matmul ( features_reshape, lstm_config.W['hidden']) + lstm_config.bias['hidden']
    features_split = tf.split(0, num_chunks, features_w)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_config.hidden_units)
    #lstm_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
    outputs, states = tf.nn.rnn(lstm_cell, features_split, dtype = tf.float32)

    return tf.matmul(outputs[-1], lstm_config.W['output']) + lstm_config.bias['output']

def load_input_data ( filename, chunk_size ) :
    x = []

    for line in open(filename) :
        a = line.replace('  ', ' ').strip().split(' ')
        t = np.array_split(a, chunk_size)
        x.append(np.array(t, dtype=np.float32))
    return np.array(x)

def load_output_data(filename) :
    y = []

    for line in open(filename) :
        a = line.replace('  ', ' ').strip().split(' ')
        y.append(a[0])

    print len(y)
    return np.array(y, dtype=np.int32) - 1

def one_hot (labels, num_classes) :

    samples = len(labels)
    new_sample = np.zeros((samples, num_classes))
    new_labels = np.array(labels)
    new_sample[np.arange(samples), new_labels] = 1

    return new_sample

def create_dir ( directory ) :

    if not os.path.exists(directory):
        print directory + ' folder dont exist'
        print 'creating directory'
        cmd = 'mkdir ' + directory
        os.system(cmd)

if __name__ == '__main__' :

    DATA_PATH = "../data/"
    DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
    MODEL_PATH = '../models/best_model'
    LOGS_PATH = '../logs/'

    create_dir('../models')
    create_dir('../logs')


    TRAIN = "train/"
    TEST = "test/"

    CHUNK_SIZE = 187
    NUM_CHUNKS = 3
    HIDDEN_UNITS = 16
    RNN_LAYERS = 1
    NUM_CLASSES = 6

    train_input_file = DATASET_PATH + TRAIN + 'X_train.txt'
    train_output_file = DATASET_PATH + TRAIN + 'y_train.txt'

    test_input_file = DATASET_PATH + TEST + 'X_test.txt'
    test_output_file = DATASET_PATH + TEST + 'y_test.txt'

    x_train = load_input_data(train_input_file, NUM_CHUNKS)
    y_train = one_hot(load_output_data(train_output_file), 6)

    x_test = load_input_data(test_input_file, NUM_CHUNKS)
    y_test = one_hot(load_output_data(test_output_file), 6)

    SAMPLES = len(x_train)

    config = lstm_configuration(NUM_CHUNKS, CHUNK_SIZE, HIDDEN_UNITS, RNN_LAYERS, SAMPLES, NUM_CLASSES)



    with tf.name_scope("input") :
        X = tf.placeholder(tf.float32, [None, config.num_chunks, config.chunk_size], name = "x_input")
        Y = tf.placeholder(tf.float32, [None, config.output_classes], name='y_input')

    with tf.name_scope('model') :
        pred_Y = lstm(X, config)

    with tf.name_scope('loss') :
        l2 = config.lambda_loss * \
            sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y)) + l2
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=config.learning_rate).minimize(cost)
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    tf.scalar_summary("loss", cost)
    tf.scalar_summary("accuracy", accuracy)
    merged_summary_op = tf.merge_all_summaries()
    #--------------------------------------------
    # step4: Hooray, now train the neural network
    #--------------------------------------------
    # Note that log_device_placement can be turned of for less console spam.
    sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    tf.initialize_all_variables().run()

    summary_writer = tf.train.SummaryWriter(LOGS_PATH, graph=tf.get_default_graph())
    best_accuracy = 0.0

    saver = tf.train.Saver(tf.all_variables())
    # Start training for each batch and loop epochs
    total_batch = int(len(x_train) / config.batch_size)
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.samples, config.batch_size),
                              range(config.batch_size, config.samples + 1, config.batch_size)):
            _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={X: x_train[start:end],
                                           Y: y_train[start:end]})

            summary_writer.add_summary(summary, i * config.batch_size + i)
        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out, correct_pred_out = sess.run([pred_Y, accuracy, cost, correct_pred], feed_dict={
                                                X: x_test, Y: y_test})

        print 'correct_pred', sum(correct_pred_out), len(correct_pred_out)
        print("training iteration: {},".format(i)+\
              " test accuracy : {},".format(accuracy_out)+\
              " loss : {}".format(loss_out))

        if accuracy_out > best_accuracy :
            saver.save(sess, MODEL_PATH)
        best_accuracy = max(best_accuracy, accuracy_out)

    print("")
    print("best accuracy: {}".format(best_accuracy))
