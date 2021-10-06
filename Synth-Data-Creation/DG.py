
from numpy.random import randint, rand
from numpy import eye
import numpy as np
import pandas as pd

def subgroups(correlated):
    """
    A function that will randomly split a group of features into subgroups that are at least of size 2.
    """
    
    subgroups = []                           #init the list for the subgroupings
    feats_left = correlated                  #init a var to keep track of how many features we have left to 'pick'
    while feats_left>0:                      #while we have some remaining
        if feats_left==2:                       #if we only have 2 remaining
            subgroups.append(2)                    #make that the last group and add to list
            feats_left=0                           #sets feats left to 0 to break out of while
        else:                                   #else
            pick = randint(2,feats_left)           #paick a random int between 2 and however many we have remaining
            feats_left -= pick                     #reduce the feats left by that amount
            if feats_left==1:                      #if there is only 1 feat left
                feats_left -=1                        #remove that last one
                pick+=1                               #and add it to the number we have picked
            subgroups.append(pick)                 #add the number we have picked to the list
    return subgroups                         #return the grouping we have created

def corrWeights(subgroups,min_c=0.5):
    """
    For the list of subgroups provided is shall randomly generate numbers whose absolute value are in the range [0.5,1) to serve as the seeding correlation value
    """
    
    if min_c<0.5 or min_c>=1:
        raise ValueError('min_c must be less than 1 and greater than 0.5')
    
    weights=[]                           #init the list to hold the correlations for each grouping
    for size in subgroups:               #for each grouping
        w = rand(size-1)-0.5                #generate random floats between [-0.5,0.5] 
        w = w*((1-min_c)/0.5)                 #scale to range we need
        w[w<0]-=min_c                         #any that are negative shift so they are in [-1,-min_c]
        w[w>0]+=min_c                         #any that are positive shift so they are in [min_c,1]
        weights.append(list(w))             #append to our list
    return weights                       #return weights

def genGroup(samples, corrFeats, corrWeight):
    """
    For the provided group details it shall create the data for the respective subgroup
    """
    
    mean = [0]*corrFeats                           #create the mean of our distributions
    cov = np.diag(np.array(corrWeight),1)          #make and off diag matrix from the correlations weights supplied
    cov += (cov.T + eye(corrFeats))                #make it a symmetric matric with 1's along the diagonal
    
    if ~np.all(np.linalg.eigvals(cov) >=0):        #if the cov matrix is not positive semidefinite
        cov = np.matmul(cov,cov)                      #turn it into one (A*A^T)
    
    #make data and put in a dataframe
    data_df = pd.DataFrame(
        np.random.multivariate_normal(
            mean,
            cov,
            samples)
    )
    return data_df

def genData(samples,groups,weights,correlated,features):
    """
    Generates the data for all the subgroups and the un-correlated features
    """
    
    master_df = pd.DataFrame()                                                                #create a master dataframe
    for i in range(0,len(groups)):                                                            #for each group
        gen_df = genGroup(samples,groups[i],weights[i])                                          #create the data
       
        #rename the columns to avoid confusion
        gen_df.rename(
            columns=dict(
                zip(
                    list(range(0,groups[i])),
                    ['f'+str(i+master_df.shape[1]) for i in list(range(0,groups[i]))])),
            inplace=True
        )
        master_df = pd.concat([master_df,gen_df],axis=1)                                       #concatenate with the master data frame
    
    for i in range(correlated,features+1):                                                     #for each of the remaining uncorrelated features
        master_df['f'+str(i)] = np.random.normal(size=(samples,1))                             #create them
    
    return master_df

def createTarget(data):
    """
    For the data created this function will generate coefficents and create the target variable
    """
    
    coeffs = (np.random.random(data.shape[1]+1)-0.5)*20                                             #generate the coefficents
    data['y'] = data.apply(lambda x: sum(x*coeffs[:-1])+coeffs[-1]+np.random.normal(0,5),axis=1)    #create the target variables
    
    return data, coeffs


def genSynthData(samples=400,features=10,correlated=5,min_c=0.5):
    """
    create a whole synthetic dataset with the structure that we require.
    """
    
    #check that the number of correlated featurs are in an acceptable range
    if (correlated<=1)|(correlated>features):
        raise ValueError("Number of correlated features must be greater than 1 and less than the number of features")
    
    groups = subgroups(correlated)                                  #create the subgroupings
    weights = corrWeights(groups,min_c)                                   #generate the correlation weights in each group
    data_df = genData(samples,groups,weights,correlated,features)   #generate the data
    data_df,coeffs = createTarget(data_df)                          #add target variable
    
    
    return data_df, groups, weights, coeffs
        
