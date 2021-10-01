
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA

class featureReduction:
    """
    A class defining the feature reduction method that we are expirmenting with.
    
    Parameter
    ---------
    threshold : float, min 0, max 1, default 0.7
                The threshold value for cutting the absolute value of the edges
                
    Attributes
    ----------
    corrMat : pandas.DataFrame
              The correlation matrix of the features
    feat_to_keep : list of int
                   The indexes of the features that are going to remain unchanged
    feat_to_reduce : list of list
                     A list of groups of features that are going to be reduced via PCA
    list_of_trained_pca : list of sklearn.decomposition.PCA
                          The list of trained PCA's
    """
    
    def __init__(self,threshold=0.7):
        """
        The init method of this class
        """
        self.threshold = threshold        #Threshold value for cutting the absolute value of the edges
        self.corrMat = None               #The correlation matrix of the features
        self.feat_to_keep = None          #The indexes of the features that are going to remain unchanged
        self.feat_to_reduce = None        #A list of groups of features that are going to be reduced via PCA
        self.list_of_trained_pca = None   #The list of trained PCA's
    
    def fit(self,train):
        """
        A function that will fit itself to the data provided
        
        Parameters
        ----------
        train : pandas.DataFrame
                the data that the PCA transformers will be trained on
        """
        
        self.corrMat = train.corr()                             #get the initial correlation matrix
        self.corrMat[abs(self.corrMat)<=self.threshold]=0       #cut the edges which are below our set threshold
        
        G = nx.from_numpy_matrix(self.corrMat.values)           #create a graph object from the correlation matrix
        self.feat_to_reduce = [list(g) for g in nx.connected_components(G) if len(g)>1]    #get all the isolated feature that shall remain unchanged
        self.feat_to_keep = [list(g)[0] for g in nx.connected_components(G) if len(g)==1]  #form the list of grouped features that are going to be reduced via pca
        
        if len(self.feat_to_reduce)!=0:                             #if we actually have groupings to reduce
            self.list_of_trained_pca = []                           #init the list that will hold the trained PCA's
            for i in range(0,len(self.feat_to_reduce)):             #for each grouping:
                current_pca = PCA(n_components=1)                   #init the pca transformer
                current_pca.fit(train.iloc[:,self.feat_to_reduce[i]])   #fit the transformer on the current grouping
                self.list_of_trained_pca.append(current_pca)            #append to the list
        else:                                                       #if not inform user
            print("No correlated features above given threshold. No PCA required.\n")
    
    def transform(self,test):
        """
        A function that will transform the data provided with the trained PCA's
        
        Parameters
        ----------
        test : pandas.DataFrame
               the data that will be transformed with the trained PCA transformers
               
        Returns
        -------
        results_df : pandas.DataFrame
                     The process data frame with both the transformed and un-transformed columns
        """
        
        if self.corrMat is not None:                #check that we actually have trained PCA
            if len(self.feat_to_reduce)!=0:         #check that we actually have grouping to reduce 
                result_df = pd.DataFrame()          #init df to hold the results

                for i in range(0,len(self.feat_to_reduce)):       #for each grouping 
                    current_pca = self.list_of_trained_pca[i]     #get the respective PCA
                    #and appened to result df
                    result_df = pd.concat([
                        result_df,pd.DataFrame(current_pca.transform(test.iloc[:,self.feat_to_reduce[i]]),columns=['pca_'+str(i)])],
                        axis=1)

                #append on features that we unchanged
                result_df = pd.concat([result_df,test.iloc[:,self.feat_to_keep].reset_index(drop=True)],axis=1)
                return result_df
            else:                                   #if there is no need for pca inform the user
                print('No transformation required.\n')
                return test
        else:                                       #if the PCA's have not being trained inform user
            raise ValueError("No PCA's trained please run .fit() or .fit_transform()")
    
    def fit_transform(data):
        """
        A function that will sequentially fit and the then transform the given data
        
        Parameters
        ----------
        data : pandas.DataFrame
               the data that the PCA transformers will be trained on and will be then subsequently transformed
        
        Returns
        -------
        results_df : pandas.DataFrame
                     The process data frame with both the transformed and un-transformed columns
        """
        
        self.fit(data)                 #fit to the data
        return self.transform(data)    #return the transformed data
    
  
        
