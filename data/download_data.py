import os

print("Checking data exist or not\n")
if not os.path.exists("UCI HAR Dataset.zip"):
    print('data downloading\n')
    cmd = 'wget http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    os.system(cmd)
    print('downloading done')

    print('extracting data....')
    cmd = 'unzip UCI\ HAR\ Dataset.zip'
    print('extracting data done')
    os.system(cmd)
else:
    print("data exist\n")

