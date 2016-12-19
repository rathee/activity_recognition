import tensorflow as tf
import numpy as np
import lstm
import sys

if __name__ == '__main__' :


    DATA_PATH = "../data/"
    DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"

    TEST_FILE_X_PATH = sys.argv[1]
    TEST_FILE_Y_PATH = sys.argv[2]

    MODEL_PATH = "../models/best_model"

    TRAIN = "train/"
    TEST = "test/"

    CHUNK_SIZE = 187
    NUM_CHUNKS = 3
    HIDDEN_UNITS = 16
    RNN_LAYERS = 1
    NUM_CLASSES = 6

    test_input_file = TEST_FILE_X_PATH
    test_output_file = TEST_FILE_Y_PATH

    x_test = lstm.load_input_data(test_input_file, NUM_CHUNKS)
    y_test = lstm.one_hot(lstm.load_output_data(test_output_file), NUM_CLASSES)

    SAMPLES = len(x_test)
    #sess.run(tf.initialize_all_variables())
    config = lstm.lstm_configuration(NUM_CHUNKS, CHUNK_SIZE, HIDDEN_UNITS, RNN_LAYERS, SAMPLES, NUM_CLASSES)

    with tf.name_scope("input") :
        X = tf.placeholder(tf.float32, [None, config.num_chunks, config.chunk_size], name = "x_input")
        Y = tf.placeholder(tf.float32, [None, config.output_classes], name='y_input')

    with tf.name_scope('model') :
        pred_Y = lstm.lstm(X, config)

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

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.initialize_all_variables() )
        saver.restore(sess, MODEL_PATH)
        print("Model restored from file: %s" % MODEL_PATH)

        pred_out, accuracy_out, loss_out, correct_pred_out = sess.run([pred_Y, accuracy, cost, correct_pred],
                                                                      feed_dict={
                                                                          X: x_test, Y: y_test})

        print("test accuracy: {}".format(accuracy_out))
