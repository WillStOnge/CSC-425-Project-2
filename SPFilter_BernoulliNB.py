import os
import math
import numpy as np

most_common_word = 3000
#avoid 0 terms in features
smooth_alpha = 1.0
class_num =2 #we have only two classes: ham and spam
class_log_prior = [0.0, 0.0] #probability for two classes
feature_log_prob = np.zeros((class_num, most_common_word)) #feature parameterized probability
SPAM = 1 #spam class label
HAM = 0 #ham class label

class BernoulliNB_class:

    #Bernoulli Naive Bayes
    def BernoulliNB(self, features, labels):
        '''//convert features to l0-norm
		/**
		 * loop over features with i and j
		 * features[i][j] > 0 ? 1 : 0;
		 */
		//calculate class_log_prior
		/**
		 * loop over labels
		 * if the value of the term in labels = 1 then ham++ 
		 * if the value of the term in labels = 0 then spam++
		 * class_log_prior[0] = Math.log(ham)
		 * class_log_prior[1] = Math.log(spam)
		 */
		//calculate feature_log_prob
		/**
		 * nested loop over features
		 * for row = features.length
		 *     for col = most_common
		 *         ham[col] + features[row][col]
		 *         spam[col] + features[row][col]
		 * for i = most_common
		 *     ham[i] + smooth_alpha
		 *     spam[i] + smooth_alpha
		 * sum of ham = files in ham + smooth_alpha*2 //difference between Multinomial and Bernoulli
		 * sum of spam = files in spam + smooth_alpha*2 //difference between Multinomial and Bernoulli
		 * for j = most_common
		 *     feature_log_prob[0] = ham[i]/sum of ham //no log here
		 *     feature_log_prob[1] = spam[i]/sum of spam //no log here
		 */'''

    #Bernoulli Naive Bayes prediction
    def BernoulliNB_predict(self, features):
       ''' //convert features to l0-norm
		/**
		 * loop over features with i and j
		 * features[i][j] > 0 ? 1 : 0;
		 */'''
        
        classes = np.zeros(len(features))

        ham_prob = 0.0
        spam_prob = 0.0
        '''/**
		 * nested loop over features with i and j
		 * calculate ham_prob and spam_prob
		 *     Math.log(feature_log_prob)*(double)features[i][j] + 
		 *         Math.log(1-feature_log_prob)*Math.abs(1-features[i][j])
		 * add ham_prob and spam_prob with class_log_prior
		 * if ham_prob > spam_prob
		 * HAM
		 * else SPAM
		 * return  classes
		 */'''

        return classes
