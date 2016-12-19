
Project structure :

activity_recognition    |
                        | - src |
                                | - my_lstm.py
                                | - test.py
                        | - data    |
                                    | - UCI HAR Dataset
                                    | - download_data.py


Requirement :

1) Python 2.7
2) Tensorflow
3) Tensorboard


Steps to follow :

* First download the data in the data folder using the command :
        python download_data.py

* Above command will "create UCI HAR Dataset" folder in the activity_recognition/data/

* For visualization, follow the below command :
        tensorboard --logdir=../logs/
                or in general sense
        tensorboard --logdir='log_directory_path'

* After above command go to web browser and type :
        localhost:6006

* Above command will show the tensorboard UI. Go to Events folder to check the accuracy and loss graph with respect to
epoch and batch size. It is showing how values are varying with each batch.
    p.s. : total values for graph representation are : num_epoches * number_of_batches_each_epoch

* Go to Graphs section to check the RNN network with each component. Go to each section and expand by pressing "+".

* To test on a new file, use below command :
        python test.py ../data/UCI\ HAR\ Dataset/test/X_test.txt ../data/UCI\ HAR\ Dataset/test/y_test.txt
                or in general case
        python test.py test_feature_file_path test_output_path

* This project is getting 95.38% accuracy on the test set.

