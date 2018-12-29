"""

You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.

Each element of train_data is a (labeled_list, feat_list) pair. 

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""

import perc
import sys, optparse, os
from collections import defaultdict

def perc_train(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    sigma = defaultdict(int)
    mistakes=0
    # For each epoch we go through all the sentences in the training set.
    for epoch in range(numepochs):
        mistakes=0
        # Each sentence is passed through the viterbi to get the argmax output of the labels.
        for sentence in train_data:
            # The prediction from viterbi is stored in pred.
            pred=perc.perc_test(feat_vec,sentence[0],sentence[1],tagset,tagset[0])
            true=[word.split()[2] for word in sentence[0]]
            count=0
            # Each label returned from the result of viterbi is checked with the true label.
            for i in range(len(pred)):
                count+=20
                if(pred[i]!=true[i]): 
                    # Record the mistakes in all epochs
                    mistakes+=1
                    # For features of each of the word in the sentence, we make the updates in weight vector.
                    for j in range(count-20,count):
                        # We give a -1 update to the features of the wrong label.
                        if (sentence[1][j],pred[i]) in feat_vec.keys():
                            feat_vec[(sentence[1][j],pred[i])]-=1
                        else:
                            feat_vec[(sentence[1][j],pred[i])]=-1
                        # We give a +1 update to the features of the true label.
                        if (sentence[1][j],true[i]) in feat_vec.keys():
                            feat_vec[(sentence[1][j],true[i])]+=1
                        else:
                            feat_vec[(sentence[1][j],true[i])]=1
                    if i>0:
                        # Similarly, we give -1 update to the wrong bigram features and 
                        # +1 update to bigram features in true label.
                        if ("B:"+pred[i-1],pred[i]) in feat_vec.keys():
                            feat_vec[("B:"+pred[i-1],pred[i])]-=1
                        else:
                            feat_vec[("B:"+pred[i-1],pred[i])]=-1
                        if ("B:"+true[i-1],true[i]) in feat_vec.keys():
                            feat_vec[("B:"+true[i-1],true[i])]+=1
                        else:
                            feat_vec[("B:"+true[i-1],true[i])]=1
            # After going through each sentence, we aggregate the weights for all the features as mentioned in 
            # http://www.cs.sfu.ca/~anoop/papers/pdf/syntax-parsing-survey-2011.pdf
            for feat,weight in feat_vec.items():
                if feat in sigma.keys():
                    sigma[feat]+=weight
                else:
                    sigma[feat]=weight
                
        print('Mistakes in epoch :',epoch,' are: ',mistakes)
        
    # We average the weight parameter using the formula γ = σ/(mT) mentioned in the above mentioned paper.
    for feat,weight in sigma.items():
        sigma[feat]=weight/(len(train_data)*numepochs)
                       
    # insert your code here
    # please limit the number of iterations of training to n iterations
    return sigma


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print("reading data ...", file=sys.stderr)
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile, verbose=False)
    print("done.", file=sys.stderr)
    feat_vec = perc_train(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)

