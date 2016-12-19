
Project structure :<br />

activity_recognition    |<br />
                        | - src |<br />
                                | - my_lstm.py<br />
                                | - test.py<br />
                        | - data    |<br />
                                    | - UCI HAR Dataset<br />
                                    | - download_data.py<br />


Requirement :<br />

1) Python 2.7<br />
2) Tensorflow<br />
3) Tensorboard<br />


Steps to follow :<br />

* First download the data in the data folder using the command :<br />
        python download_data.py<br />

* Above command will "create UCI HAR Dataset" folder in the activity_recognition/data/ <br />

* For visualization, follow the below command :<br />
        tensorboard --logdir=../logs/  <br />
                or in general sense  <br />
        tensorboard --logdir='log_directory_path'  <br />

* After above command go to web browser and type : <br />
        localhost:6006  <br />

* Above command will show the tensorboard UI. Go to Events folder to check the accuracy and loss graph with respect to
epoch and batch size. It is showing how values are varying with each batch.  <br />
    p.s. : total values for graph representation are : num_epoches * number_of_batches_each_epoch  <br />

* Go to Graphs section to check the RNN network with each component. Go to each section and expand by pressing "+". <br />

* To test on a new file, use below command :  <br />
        python test.py ../data/UCI\ HAR\ Dataset/test/X_test.txt ../data/UCI\ HAR\ Dataset/test/y_test.txt  <br />
                or in general case  <br />
        python test.py test_feature_file_path test_output_path  <br />

* This project is getting 95.38% accuracy on the test set.  <br />

