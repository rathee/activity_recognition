Project structure :<br />

activity\_recognition |<br /> | - src |<br /> | - my\_lstm.py<br /> | -
test.py<br /> | - data |<br /> | - UCI HAR Dataset<br /> | -
download\_data.py<br />

Requirement :<br />

1)  Python 2.7<br />
2)  Tensorflow<br />
3)  Tensorboard<br />

Steps to follow :<br />

-   First download the data in the data folder using the command :<br />
    python download\_data.py<br />

-   Above command will "create UCI HAR Dataset" folder in the
    activity\_recognition/data/ <br />

-   For visualization, follow the below command :<br /> tensorboard
    --logdir=../logs/ <br /> or in general sense <br /> tensorboard
    --logdir='log\_directory\_path' <br />

-   After above command go to web browser and type : <br />
    localhost:6006 <br />

-   Above command will show the tensorboard UI. Go to Events folder to
    check the accuracy and loss graph with respect to epoch and batch
    size. It is showing how values are varying with each batch. <br />
    p.s. : total values for graph representation are : num\_epoches \*
    number\_of\_batches\_each\_epoch <br />

-   Go to Graphs section to check the RNN network with each component.
    Go to each section and expand by pressing "+". <br />

-   To test on a new file, use below command : <br /> python test.py
    ../data/UCI HAR Dataset/test/X\_test.txt
    ../data/UCI HAR Dataset/test/y\_test.txt <br /> or in general case
    <br /> python test.py test\_feature\_file\_path test\_output\_path
    <br />

-   This project is getting 95.38% accuracy on the test set. <br />


