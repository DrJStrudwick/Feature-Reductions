# Feature-Reductions
 
This repo is to test & develop my hypothesis for using PCA to perform feature reduction as a pre-procerssing step for model creations.

## Concept

Imagine you have a collection of input features that are to be used to build a ML model, let us assume Linear Regression (LR). Now suppose that some of the input features are correlated. This would means that if you have one of the features you can create a bijective mapping that you take you from one to the other (approximately). If you then proceeded to build a model using these features then there would be a lot of redundant information in the system, potentially increasing training time and reducing accuracy.

Thinking in terms of linear algebra, our input features are our basis vectors. What we desire is to have our input features be the equivalent of a linearly independent set. If one or more of the input features is correlated to others and can be expressed in terms of others then they are not an independant set.

## proposed FR method

For the time being we shall assume that all the input features are continous variables. Adapatation/extension to Discreet-continous and Discreet-Discreet interactions are still being worked upon.

This is the proceedure we propose. Once the data cleaning has been completed, produce a feature correlation matrix using a metric of you choice. We shall use Pearson/Spearman correlation. If we interpret this matrix as an adjaceny matrix then we can visualise it as a graph with weighted edges. Next choose some threshold $\tau$ that shall be the level above which we shall class to features to be related. Any edge in our graph whose absolute value is less than the chosen threshold set to 0. At this point the graph should form several disjoin components, and there maybe nodes (features) that are isolated and not connected to anything.

For the components that are not isolated nodes we take the corresponding features from the data set and we use PCA to reduce those down to single dimension. This is then used for the model creation inplace of the features that made it.

## validation expirement

To test/validate our method we shall run the following expirement.

We shall construct a synthetic data set, so we pocess a underlying ground truth which we can make comparisions against. We shall produce 3 models.
 1. Natural model - this shall be a model with no PCA performed using all of the input features to create the model.
 2. Hybrid model - this shall be the model where we implement our proposed FR method to create the model.
 3. PCA model - this shall be a model where we apply PCA to all of the features regardless of their correlations and use the top $k$ princepal components.


## Synthetic Data creation

