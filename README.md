# DomainDiscovery

This machine learning tool is to rank a newly discovered website by relevance. 

## Setup

###First, install <a href="https://www.tensorflow.org/">TensorFlow</a> and <a href="https://github.com/tensorflow/skflow/tree/master/skflow">SKFlow </a>

###Second, prepare training set
Put highly relevant websites into seedDomains.txt
Put mildly relevant websites into visitedDomains.txt
Put slightly relevant websites into linkingDomains.txt
Run cp.generateTrainData()

###Third, prepare target websites
Put the target websites into test.txt
Run cp.generateTestData()

###Finally, train and run the model.
c = Classifier()
c.train()
c.predict()
