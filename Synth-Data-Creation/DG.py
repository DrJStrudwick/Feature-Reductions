
from numpy.random import randint, rand
from numpy import eye
import numpy as np
import pandas as pd

def subgroups(correlated):
    subgroups = []
    feats_left = correlated
    while feats_left>0:
        if feats_left==2:
            subgroups.append(2)
            feats_left=0
        else:
            pick = randint(2,feats_left)
            feats_left -= pick
            if feats_left==1:
                feats_left -=1
                pick+=1
            subgroups.append(pick)
    return subgroups

def corrWeights(subgroups):
    weights=[]
    for size in subgroups:
        w = rand(size-1)-0.5
        w[w<0]-=0.5
        w[w>0]+=0.5
        weights.append(list(w))
    return weights

def genGroup(samples, corrFeats, corrWeight):
    mean = [0]*corrFeats
    cov = np.diag(np.array(corrWeight),1)
    cov += (cov.T + eye(corrFeats))
    
    if ~np.all(np.linalg.eigvals(cov) >=0):
        cov = np.matmul(cov,cov)
    
    data_df = pd.DataFrame(
        np.random.multivariate_normal(
            mean,
            cov,
            samples)
    )
    return data_df

def genData(samples,groups,weights,correlated,features):
    master_df = pd.DataFrame()
    for i in range(0,len(groups)):
        gen_df = genGroup(samples,groups[i],weights[i])
        gen_df.rename(
            columns=dict(
                zip(
                    list(range(0,groups[i])),
                    ['f'+str(i+master_df.shape[1]) for i in list(range(0,groups[i]))])),
            inplace=True
        )
        master_df = pd.concat([master_df,gen_df],axis=1)
    
    for i in range(correlated,features+1):
        master_df['f'+str(i)] = np.random.normal(size=(samples,1))
    
    return master_df

def createTarget(data):
    coeffs = (np.random.random(data.shape[1]+1)-0.5)*20
    data['y'] = data.apply(lambda x: sum(x*coeffs[:-1])+coeffs[-1]+np.random.normal(0,5),axis=1)
    
    return data, coeffs


def genSynthData(samples=400,features=10,correlated=5):
    if (correlated<=1)|(correlated>features):
        raise ValueError("Number of correlated features must be greater than 1 and less than the number of features")
    
    groups = subgroups(correlated)
    weights = corrWeights(groups)
    data_df = genData(samples,groups,weights,correlated,features)
    data_df,coeffs = createTarget(data_df)
    
    
    return data_df, groups, weights, coeffs
        
