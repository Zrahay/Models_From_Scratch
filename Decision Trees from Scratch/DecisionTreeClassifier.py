import numpy as np
import pandas as pd
import random


"""
This a Decison Tree Classifier which works on the cart algorithm, i.e., Gini Impurity
"""

def train_test_split(df, test_size):
    # We are first going to check if the input test size is a proportion or is it some kind of real/integer value
    if isinstance(test_size, float):
        test_size = test_size * len(df)
    
    indices = df.index.tolist() # This gives us a list named indices which consists of all the indexes present in the dataframe. So, for example if we have about 2000 data samples, then we will get 0 to 1999 in the list of indices
    test_indices = random.sample(indices, k = test_size)
    
    df_test = df.loc[test_indices]
    df_train = df.drop(test_indices)

    return df_train, df_test








class DecisionTreeClassifier():

    def __init__(self, max_depth = None, criteria = 'gini'):
        self.max_depth = max_depth
        self.criteria = criteria

    # Now we can define the helper functions which are either going to be private or they are going to be public

    # The probable helper functions in this case are going to be gini_calculation, best_split, etc

    def __gini(left_split, right_split): # We need the data to be in the form of lists instead of Pandas series so we can do the handling here itself
        """
        This function has the implementation of calculating the gini impurity. Considering we have two lists which replicate the 2 rows that we will be using
        """
        
        # We need to calculate the Gini Imppurity of these values

        one_left = 0
        zero_left = 0
        zero_right = 0
        one_right = 0

        for val in left_split:
            
            if val == 1:
                one_left += 1
            else:
                zero_left += 1
        
        for val in right_split:

            if val == 1:
                one_right += 1
            else:
                zero_right += 1

        
        # Now we can use the formula of Gini Impurity as (1 - (one**2 + zero**2))
        ig_left = 1 - ((one_left / len(left_split))**2  + (zero_left / (len(left_split)))**2)

        ig_right = 1 - ((one_right / len(right_split))**2 + (zero_right / len(right_split))**2)

        total_len = len(left_split) + len(right_split)

        weighted_gini = (len(left_split)/total_len) * ig_left + (len(right_split)/total_len) * ig_right

        return weighted_gini
    



    def __information_gain(gini_org = 0.0, left_split = None, gini_left = 0.0, right_split = None, gini_right = 0.0):
        """
        This function is used to calculate the Information gain that a particular split produces
        """
        total_len = len(left_split) + len(right_split)

        ig = gini_org - ((len(left_split) * gini_left) / (total_len) + (len(right_split) * gini_right) / (total_len))

        return ig
    

    def __best_split(self, left_split = None, right_split = None, map = None):
        """
        This function decides the best split based on the Gini Impurity and the best Information Gain. 
        """

        # We have the first feature values inside left_split and second feature values inside right_split.

        # Step 1 is to get the initial Gini Impurity for each of the list values

        label_left = []

        for val in left_split:
            label = map[val]
            label_left.append(label)
        
        label_right = []
        for val in right_split:
            label = map[val]
            label_right.append(label)

        # Now we need to find the medians of the adjacent values and so, for that we need to sort both the lists

        left_split.sort()
        right_split.sort()

        median_left = []

        i = 0
        j = i + 1

        while j < len(left_split):
            median_val = (left_split[i] + left_split[j]) / (2.0)
            median_left.append(median_val)
            i += 1
            j += 1
        
        median_right = []
        i = 0
        j = i + 1
        while j < len(right_split):
            median_val = (right_split[i] + right_split[j]) / (2.0)
            median_right.append(median_val)
            i += 1
            j += 1

        # After getting medians for both left and right, we can take each median and calculate the weighted gini impurity because of that. 

        # SO, we can start with the median_left list
        information_gain_left = []
        for threshold in median_left:
            # For each threshold, we need to calculate the original Gini Impurity for the list that we have right now

            weighted_original = self.__gini(label_left, label_right)
            new_left_left_split = []
            new_left_right_split = []
            for i in left_split:
                if i <= threshold:
                    new_left_left_split.append(i)
                else:
                    new_left_right_split.append(i)
            
            # Now we need the weighted gini impurity to be calculated for this left split and right split based on the threshold
            # For these weighted ginis, we need the labels corresponding to the left and right split
            left_labels = []
            right_labels = []

            for val in new_left_left_split:
                left_labels.append(map[val])

            for val in new_left_right_split:
                right_labels.append(map[val])
            
            weighted_new = self.__gini(left_labels, right_labels)
            ig = self.__information_gain(weighted_original, weighted_new)
            information_gain_left.append((ig, threshold))
        

        information_gain_right = []
        for threshold in median_right:
            weighted_original = self.__gini(label_left, label_right)
            new_right_left_split = []
            new_right_right_split = []
            for i in right_split:
                if i <= threshold:
                    new_right_left_split.append(i)
                else:
                    new_right_right_split.append(i)
            
            left_labels = []
            right_labels = []

            for val in new_right_left_split:
                left_labels.append(map[val])
            
            for val in new_right_right_split:
                right_labels.append(map[val])
            

            weighted_new = self.__gini(left_labels, right_labels)
            ig = self.__information_gain(weighted_original, weighted_new)
            information_gain_right.append((ig, threshold))
        

            

            







        
    





      

    
        

