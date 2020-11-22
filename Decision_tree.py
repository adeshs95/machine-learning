import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import random
from pprint import pprint
dataset=pd.read_csv("Breast Cancer_dataset.csv")

def divide(dataset,test_size):#split data into 80/20
    if isinstance(test_size,float):
        test_size=round(test_size*len(dataset))
    indices=dataset.index.tolist() # Get indices into list
    test_indices=random.sample(population=indices,k=test_size)#gatting random indices
    test_set=dataset.loc[test_indices]  #Getting rows corresponding to indices
    train_set=dataset.drop(test_indices)
    return test_set,train_set

def classify(data_subset):  #To get the most occured class value
    return find_most_common_value(data_subset,-1)#majority class value in data_subset


def pure(data_subset):#checking where all the instances in dataset are pure or not
    label_col=data_subset[:,-1]
    unique_classes=np.unique(label_col)
    if len(unique_classes)==1:
        return 1
    else:
        return 0

def entropy(data_subset):
    label_col=data_subset[:,-1]
    _,count=np.unique(label_col,return_counts=True)
    prob=count/count.sum()
    entropy=sum(-prob*np.log2(prob))
    return entropy


def overall_entropy(data_left,data_right):
   Total_instances=len(data_left)+len(data_right)
   pl=len(data_left)/Total_instances
   pr=len(data_right)/Total_instances
   total_entropy=pl*entropy(data_left)+pr*entropy(data_right)
   return total_entropy

def potential_attribute(data_subset):#Returns the potential attributes with their unique values

    potential_attributes={}
    _,no_attributes=data_subset.shape
    for i in range(0,no_attributes-1):
        unique_values=np.unique(data_subset[:,i])
        potential_attributes[i]=unique_values
    return potential_attributes

def find_most_common_value(dataset,column_index):#Find most common class value
    column=dataset[:,column_index]
    unique_values,count=np.unique(column,return_counts=True)
    index=count.argmax()
    return unique_values[index]
    
def split_tree(data_subset,split_attribute,split_value):#Splits the tree
    split_attribute_value=data_subset[: , split_attribute]
    left_instances=data_subset[split_attribute_value==split_value]
    right_instances=data_subset[split_attribute_value!=split_value]
    return left_instances,right_instances

def Find_best_attribute(data_subset,potential_attributes):#Find the best attribute for spilliting
    min_Entropy=1000
    
    for column_index in potential_attributes:
        for value in potential_attributes[column_index]:
            if value=='?':  #handling missing values
                value=find_most_common_value(data_subset,column_index)
            data_left,data_right=split_tree(data_subset,column_index,value)
            current_entropy=overall_entropy(data_left,data_right)
            if min_Entropy>current_entropy:
                min_Entropy=current_entropy
                best_attribute=column_index
                best_value=value
    return best_attribute,best_value



def build_tree(dataframe,count,min_samples, max_depth):#Recursively builds the decisioin tree

    if not count:
        global attribute_names
        attribute_names = dataframe.columns
        dataset=dataframe.values
        
    else:
        dataset = dataframe
    if pure(dataset) or len(dataset)<min_samples or count==max_depth:#terminating conditions
        return classify(dataset)

    count += 1
    potential_splits=potential_attribute(dataset)
    split_attribute,split_value=Find_best_attribute(dataset,potential_splits)
    data_left,data_right=split_tree(dataset,split_attribute,split_value)

    if len(data_left) ==0 or len(data_right) ==0:
        return classify(dataset)

    column_name = attribute_names[split_attribute]
    Node= "{} = {}".format(column_name,split_value)
    subtree = {Node: []}

    true = build_tree(data_left,count,min_samples,max_depth)
    false = build_tree(data_right,count,min_samples,max_depth)

    if true==false:
        subtree=true

    else:
        subtree[Node].append(true)
        subtree[Node].append(false)

    return subtree

def classify_example(example,tree):#Classifies the given instance as recurring or non recurring event
    if not isinstance(tree, dict):
        return tree
    Node=list(tree.keys())[0]

    attribute_name, _, value=Node.split()
    if str(example[attribute_name])==value:
        ans=tree[Node][0]
    else:
        ans=tree[Node][1]
    if not isinstance(ans,dict):
        return ans
    else:
        return classify_example(example,ans)

def do_predictions(df,tree):
    
    if len(df) != 0:
        predictions=df.apply(classify_example, args=(tree,), axis=1)
    else:
        # "df.apply()"" with empty dataframe returns an empty dataframe,
        # but "predictions" should be a series instead
        predictions = pd.Series()
        
    return predictions

def calc_accuracy(dataset,tree):#Calculates the accuracy of decision tree for the given dataset
    dataset["classication"]=do_predictions(dataset,tree)
    dataset["check"]=dataset["classication"]==dataset["Label"]
    accuracy=dataset["check"].mean()
    return accuracy

def Q1(dataset,max_depth):#averaging over 10 random 80/20 splits with depth from 2 to 11
    
    max_accuracy=0
    for i in  range(2,11):
        test_set,training_set=divide(dataset,0.2)
        tree=build_tree(training_set,0,2,max_depth)
        accuracy=calc_accuracy(test_set,tree)
        if max_accuracy<accuracy:
            max_accuracy=accuracy
            Best_tree=tree
            no=i

    return Best_tree,max_accuracy#test_set=dataset.loc[test_indices] 
        
    


def Q2(dataset):#Finding best possible depth limit for the decision tree
    test_accuracy=list()
    max_accuracy=0

    for depth in range(3,13):
        tree,accuracy =  Q1(dataset,depth)
        test_accuracy.append(accuracy)
        
        if max_accuracy<accuracy:
            max_accuracy=accuracy
            Best_tree=tree
            best_depth=depth

    depth=list(range(3,13))
    plot_graph(depth,test_accuracy)
    return Best_tree,best_depth

def plot_graph(depth,test_accuracy):#Plots the graph
    plt.plot(depth,test_accuracy) 
  

    plt.xlabel('depth') 

    plt.ylabel('accuracy') 
  

    plt.title('Plot for Questioin 2!') 
  

    plt.show() 
    

def filter_dataset(dataset,Node):#Perform filtering to the dataset
    attribute,_,value =Node.split()
    dataset_left = dataset[dataset[attribute].astype(str) == value]
    dataset_right=dataset[dataset[attribute].astype(str) != value]
    
    return dataset_left,dataset_right

def determine_errors(valid_set,tree):#Determine errors in predictions of decisiion tree
    predictions=do_predictions(valid_set,tree)
    actual_values =valid_set.Label
    return sum(predictions != actual_values)

def pruning_conse(tree,train_set,valid_set):#check whether the subtree should be prune or not
    leaf=classify(train_set)
    pruned_errors = determine_errors(valid_set,leaf)
    Not_pruned_errors= determine_errors(valid_set,tree)

    if pruned_errors<=Not_pruned_errors:
        return leaf
    else:
        return tree

def post_pruning(train_set,valid_set,tree):#perform post pruning
    Node=list(tree.keys())[0]
    
    print(Node,"a")
    print(tree[Node])
    true,false=tree[Node]
    if not isinstance(true,dict) and not isinstance(false,dict):
        return pruning_conse(tree,train_set,valid_set)
    else:
        train_set_left, train_set_right = filter_dataset(train_set,Node)
        valid_set_left, valid_set_right = filter_dataset(valid_set,Node)
        
        if isinstance(true,dict):
            true= post_pruning(true,train_set_left,valid_set_left)
            
        if isinstance(false,dict):
            false= post_pruning(false,train_set_right,valid_set_right)
        
        tree={Node:[true,false]}
    
        return pruning_conse(tree,train_set,valid_set)


def Q3(tree,dataset):##Performing post pruning
    test_set,training_set=divide(dataset,0.2)
    valid_set,train_set=divide(training_set,0.2)
    print(len(valid_set))
    pruned_tree=post_pruning(train_set,valid_set,tree)
    pruned_accuracy=calc_accuracy(test_set,pruned_tree)
    accuracy=calc_accuracy(test_set,tree)



max_depth = int(input("Enter the depth:"))
Best_tree_q1,max_accuracy_q1=Q1(dataset,max_depth)
pprint(Best_tree_q1,width=50) 
Best_tree,best_depth=Q2(dataset)
Best pruned_tree=Q3(Best_tree,dataset)  
pprint(Best_tree,width=50)  #Priniting the decision tree got in Q2 for Q4











        


    










