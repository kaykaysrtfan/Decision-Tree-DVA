#Karthik Kannan

import csv
import numpy as np
from math import log
import operator

myname = "Karthik-Kannan-"


#Random number to take values from dataset
J=262 

#Getting the features and values of each row from the dataset
def get_data(training_file):
    dataset = []
    for index,line in enumerate(open(training_file,'rb').readlines()):
        line = line.strip()
        element = line.split(',')
        dataset.append([(element[i]) for i in range(len(element)-1)]+[element[len(element)-1]])

    nodes = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

    return dataset,nodes

#Splitting the dataset to get right and left subtree(along with nodes)
def data_split(dataset,feature_index,labels):

    left_data = []
    right_tree = []
    left_label = []
    right_label = []
    datasets = []
    for data in dataset:
        datasets.append(data[0:11])

    count = 0
    avg = 0
    for index in range(1, len(datasets)):
        count+=1
        avg += float(datasets[index][feature_index])

    
    avg = avg/count 
    mean_value = avg

    for index in range(1,len(dataset)):
        if float(dataset[index][feature_index]) > mean_value:
            right_tree.append(dataset[index])
            right_label.append(dataset[index][-1])
        else:
            left_data.append(dataset[index])
            left_label.append(dataset[index][-1])

    return left_data,right_tree,left_label,right_label

#Calculating the entropy value
def find_entropy(dataset):

    n = len(dataset)    
    aggregate_label = {}
    for data in dataset:
        label = data[-1]
        if aggregate_label.has_key(label):
            aggregate_label[label] += 1
        else:
            aggregate_label[label] = 1
    entropy = 0
    for label in aggregate_label:
        prob = float(aggregate_label[label])/n
        entropy -= prob*log(prob,2)
    return entropy

#Calculating the mean of a dataset
def calc_mean(dataset, feature_index):
    count = 0
    avg = 0.0
    for index in range(1, len(dataset)):
        count+=1
        avg += float(dataset[index][feature_index])

    avg = avg/float(count)
    #print "avg: ",avg
    return avg

#Information Gain = IG
def calculate_IG(dataset,feature_index,base_entropy):

    datasets = []

    mean_value = calc_mean(dataset,feature_index)   
    left_data = []
    right_tree = []
    for index in range(1,len(dataset)):
        if float(dataset[index][feature_index]) > mean_value:
            right_tree.append(dataset[index])
        else:
            left_data.append(dataset[index])
    condition_entropy = float(len(left_data))/len(dataset)*find_entropy(left_data) + float(len(right_tree))/len(dataset)*find_entropy(right_tree)
    return base_entropy - condition_entropy 

#Calculating the gain ratio
def calculate_IG_ratio(dataset,feature_index):

    base_entropy = find_entropy(dataset)

    info_gain = calculate_IG(dataset,feature_index,base_entropy)
    info_gain_ratio = info_gain/base_entropy
    return info_gain_ratio

#Finding the best node to split upon
def find_best_node(dataset,nodes):

    split_index_node = -1
    max_info_gain_ratio = 0.0
    for i in range(len(nodes)):
        info_gain_ratio = calculate_IG_ratio(dataset,i)
        if info_gain_ratio > max_info_gain_ratio:
            max_info_gain_ratio = info_gain_ratio
            split_index_node = i
    #print "index to split: ", split_index_node
    return split_index_node

#returning the label 
def find_best_label(labels):

    aggregate_label = {}
    for label in labels:
            if label not in aggregate_label.keys():
                aggregate_label[label] = 1
            else:
                aggregate_label[label] += 1
    sorted_aggregate_label = sorted(aggregate_label.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sorted_aggregate_label[0][0]

#Decision tree creation function
def create_dec_tree(dataset,labels,nodes):

    if len(labels) == 0:
        return 'NULL'
    if len(labels) == len(labels[0]):
        return labels[0]
    if len(nodes) == 0:
        return find_best_label(labels)
    if find_entropy(dataset) == 0:
        return find_best_label(labels)
    split_node_index = find_best_node(dataset,nodes)
    split_node = nodes[split_node_index]
    decision_tree = {split_node:{}}
    #Steps for alternate implementation
    #if calculate_IG_ratio(dataset,split_node_index) < 0.05:     
    #    return find_best_label(labels)
    del(nodes[split_node_index])
    left_data,right_tree,labels_less,labels_greater = data_split(dataset,split_node_index,labels)

    left_data.insert(0,nodes)

    right_tree.insert(0,nodes)

    decision_tree[split_node]['<='] = create_dec_tree(left_data,labels_less,nodes)
    decision_tree[split_node]['>'] = create_dec_tree(right_tree,labels_greater,nodes)

    return decision_tree

#Calculate mean for whole dataset
def get_means(train_dataset):

    dataset = []
    count = 0
    avg = [0,0,0,0,0,0,0,0,0,0,0]
    for index in range(1, len(train_dataset)):
        count+=1
        for k in range(0,11):
            avg[k] += float(train_dataset[index][k])

    for i in range(0,11):
        avg[i] = avg[i]/count 
    return avg

#Classify the testing dataset
def classifier(decision_tree,nodes,test_data,mean_values):
    
    first_fea = decision_tree.keys()[0]

    index_node = nodes.index(first_fea)

    if float(test_data[index_node]) <= mean_values[index_node]:
        sub_tree = decision_tree[first_fea]['<=']
        if type(sub_tree) == dict:
            return classifier(sub_tree,nodes,test_data,mean_values)
        else:
            return sub_tree
    else:
        sub_tree = decision_tree[first_fea]['>']
        #print sub_tree
        if type(sub_tree) == dict:
            return classifier(sub_tree,nodes,test_data,mean_values)
        else:
            return sub_tree


#Get values of each training indicator
def retreive_values(training_file):
    labels = []
    for index,line in enumerate(open(training_file,'rU').readlines()):
        each = line.split(',')[-1]
        labels.append(each)
    #print labels[0]
    return labels


#Main module
def run_decision_tree():

    final_accuracy = 0.0
    accuracy_10 = [0,0,0,0,0,0,0,0,0,0]

    # Load data set
    with open("hw4-data.csv") as f:
        next(f, None)
        data = [line for line in csv.reader(f, delimiter=",")]
    print "Number of records: %d" % len(data)
    # Split training/test sets
    # You need to modify the following code for cross validation.
    for K in range(1,11):
        training_set = [x for i, x in enumerate(data) if i % J != K]
        test_set = [x for i, x in enumerate(data) if i % J == K]

        avg = [0,0,0,0,0,0,0,0,0,0,0]
        count = 0
        for each in training_set:
            count +=1
            for i in range(0,11):
                avg[i] += 0.0 + float(each[i])

        for i in range(0, len(avg)):     
            avg[i] = avg[i]/float(count)

        for i, s in enumerate(training_set):
            for j in range(0,11): 
                training_set[i][j] = float(s[j]) 

        for each in training_set:
            for i in range(0,11):
                if float(each[i])>avg[i]:
                    each[i] = float(1.0)
                else:
                    each[i] = float(0.0)

        nodes = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
        with open('new_training.csv', 'wb') as csvfile:
            file_write = csv.writer(csvfile)
            file_write.writerow(nodes)
            for j in range(0,count):
                file_write.writerow(training_set[j])

        with open('new_testing.csv', 'wb') as csvfile:
            file_write = csv.writer(csvfile)
            file_write.writerow(nodes)
            for j in range(0,len(test_set)):
                file_write.writerow(test_set[j])
        decision_tree = []
        labels = retreive_values('new_training.csv')
        training_file = 'new_training.csv'
        testing_file = 'new_testing.csv'
        train_dataset,train_nodes = get_data(training_file)
        decision_tree = create_dec_tree(train_dataset,labels,train_nodes)

        mean_values = get_means(train_dataset)
        test_dataset,test_nodes = get_data(testing_file)

        n = len(test_dataset)
        correct = 0

        for test_data in test_dataset[1:]:
            label = classifier(decision_tree,test_nodes,test_data,mean_values)
            if label == test_data[-1]:
                correct += 1
        #print "accuracy: ",correct/float(n)
        accuracy_10[K-1] = correct/float(n)

    #print accuracy_10
    final_accuracy = reduce(lambda x, y: x + y, accuracy_10) / len(accuracy_10)
    #print "final_accuracy: ",(final_accuracy)

    # Accuracy
    print "Accuracy: %.4f" % final_accuracy       
    

    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("Accuracy: %.4f" % final_accuracy)
    f.close()



if __name__ == '__main__':
    run_decision_tree()
