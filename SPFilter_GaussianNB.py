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

#mean and standard deviation for Gaussian distribution
mean = np.zeros((class_num, most_common_word))
std = np.zeros((class_num, most_common_word))

class GaussianNB_class:
    #Gaussian Naive Bayes 
    def GaussianNB(self, features, labels):
        '''//calculate mean
		/**
		 * for i in most_common
		 *     for j in features.length
		 *         sum_ham +=features[j][i];
		 *         sum_spam +=features[j][i];
		 *     mean[0] = sum_ham / sum of ham files in labels
		 *     mean[1] = sum_spam / sum of spam files in labels
		 */
		//calculate standard deviation
		/**
		 * for i in most_common
		 *     for j in features.length
		 *         seq_ham +=Math.pow(features[j][i]-mean[0][i], 2);
		 *         seq_spam +=Math.pow(features[j][i]-mean[1][i], 2);
		 *      std[0] = Math.sqrt(seq_ham/sum of ham files in labels);
		 *      std[1] = Math.sqrt(seq_spam/sum of spam files in labels);
		 */'''

    #Gaussian Naive Bayes prediction
    def GaussianNB_predict(self, features):
        classes = np.zeros(len(features))

        ham_prob = 0.0
        spam_prob = 0.0
        '''//calculate the Gaussian value for each feature
             and summ over one specific file
		/**
		 * nested loop over features with i and j
		 * calculate ham_prob and spam_prob
		 * 1.0/(std*Math.sqrt(2.0*Math.PI))*
		   Math.exp(-(Math.pow((features[i][j]-mean), 2)/2.0*Math.pow(std, 2)));
		 * if ham_prob > spam_prob
		 * HAM
		 * else SPAM
		 * return  classes
		 */'''

        return classes
