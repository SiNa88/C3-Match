import numpy as np
import time
import os
import sys
sys.path.append('./helpers')

start_time = time.monotonic()
training_csv_file = 'train.csv' 
testing_csv_file = 'test.csv' 
#print (training_csv_file)



from helpers import read_amazon_csv
text_train_all, target_train_all = read_amazon_csv(training_csv_file, max_count=100000)
##for text in text_train_all[:3]:
##	print(text + "\n")
#print(target_train_all[:3])

##for text in text_train_all[-3:]:
##	print(text + "\n")
#print(target_train_all[-3:])


#from sklearn.model_selection import train_test_split

#text_train_small, text_validation, target_train_small, target_validation = train_test_split(
#    text_train_all, np.array(target_train_all), test_size=.5, random_state=42)

text_test_all, target_test_all = read_amazon_csv(testing_csv_file)
##print (len(text_test_all), len(target_test_all))


from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline


'''# Define a pipeline to search for the best combination of 
# and classifier regularization.
h_vectorizer = HashingVectorizer(encoding='latin-1')
h_pipeline = Pipeline((('vec', h_vectorizer), ('clf', PassiveAggressiveClassifier(C=1, warm_start=True))))
prediction0 = h_pipeline.fit(text_train_all, target_train_all).score(text_test_all, target_test_all)
print (prediction0)'''

vocabulary_vec = TfidfVectorizer(encoding='latin-1', use_idf=False)
'''vocabulary_pipeline1 = Pipeline((('vec', vocabulary_vec),('clf', PassiveAggressiveClassifier(C=1, warm_start=True))))
prediction1 = vocabulary_pipeline1.fit(text_train_all, target_train_all).score(text_test_all, target_test_all) #.score(text_validation, target_validation)
print (prediction1)'''

# set the tolerance to a large value to make the example faster
###lr = LogisticRegression(maxIter=10, regParam=0.01)
logistic = LogisticRegression(max_iter=100000000, tol=0.001)
vocabulary_pipeline2 = Pipeline(steps=[('vec', vocabulary_vec), ("logistic", logistic)])
prediction2 = vocabulary_pipeline2.fit(text_train_all, target_train_all).score(text_test_all, target_test_all) #.score(text_validation, target_validation)
print (prediction2)
#print ((vocabulary_vec.vocabulary_))

elapsed_time = time.monotonic() - start_time
print("Done - time:{}".format(elapsed_time))

