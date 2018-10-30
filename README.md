# Distributed_Multiclass_multilabel_Logistic_Regression
This Repository Contain both Local and Distributed implimentation of Multiclass Multilabel Logistic Regression
1. Local_Logistic_Regression
  This file contains code for local implementation of Logistic Regression which trains model and returns test and train  accuracy.
2. Distributed_Logistic_Regression
This file contains code for Tensorflow based implementation of Logistic Regression using pickled parameter files from the local execution for time saving due to similar code.
3. Syncronous_SGD.py : This file contains code for execution of Distributed Synchronous execution of Logistic Regression.

4 .Asyncronous_SGD.py: This file contains code for execution of Distributed Asynchronous execution of Logistic Regression.

5. Stall_syncronous.py: This file contains code for execution of Stall syncronous Distributed Asynchronous execution of Logistic Regression.
# References 
  This code was build upon the code provided on the tensorflow website for distributed implimentation.
  https://www.tensorflow.org/deploy/distributed
  
