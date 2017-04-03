# Imprecise Commands, Version 1, Prototype 1
#
# This copyright header adapted from the Linux 4.1 kernel.
#
# Copyright 2017 Paul Lu
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#      
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Original framework from sentdex on YouTube, and nltk.org
#	Credit to:  Harrison Kinsley, http://sentdex.com/about/author/
# Code has been changed substantially since.
# See also:  https://gist.github.com/Azure-rong/9754631

import sys
import nltk
import random
# import pdb	# pdb.set_trace()
from nltk.probability import DictionaryProbDist

# TODO:  Should use command-line arguments too
verbose = False
singleSentenceTest = True
max_print_comment_lines = 2

# Training vs. testing:  Value between 0.0 and 1.0
fold_split_percent = 0.8

# Number of features to use:  3000 was the start
num_features = 200

ngrams = []
current_class = "None"
fulltraining = []
fullsentence = []

# ********************************************************
# Only find features that are True
def true_features(features_string_document, word_features_all):
    wordsInThisString = set(features_string_document)
    features = {}
    for w in wordsInThisString:	# Only True features
        if (w in word_features_all) and ( features_string_document[w] == True ):
            features[w] = True
    return features


# Only find features that are in the string being classified
def features_in_string(features_string_document, word_features_all):
    wordsInThisString = set(features_string_document)
    features = {}
    for w in wordsInThisString:	# Only look at features in use, either T or F
        features[w] = (w in word_features_all)
    return features


# Answer T/F for all possible features from the training data
def find_features(features_string_document, word_features_all):
    wordsInThisString = set(features_string_document)
    features = {}
    for w in word_features_all:	# Record True/False for all possible features
        features[w] = (w in wordsInThisString)
    return features

# TODO:  Need to fix this citation to stackoverflow, per Abram Hindle's advice
# http://stackoverflow.com/questions/10472907/how-to-convert-dictionary-into-string
# User=gnr, May 6, 2012
def dict_to_string( d ):
    s= ''.join('\'{}\':{} '.format(key, val) for key, val in d.items())
    return( s )

def dict_to_string_nl( d ):
    s= ''.join('\'{}\':{}\n'.format(key, val) for key, val in d.items())
    return( s )

def classify_tuple_with_debug( inWords, word_features_all ):
    print( "InWords: ", inWords )

    inFeatures = find_features( inWords, word_features_all )
    # print( "inFeatures:", len(inFeatures), dict_to_string( inFeatures ) )
    dist = classifier.prob_classify( inFeatures )
    predClass = dist.max()
#
    dString = "(len InWords: " + str(len(inWords)) + ") "
    dString += dict_to_string( features_in_string(inWords, word_features_all) ) + " -- Prob: "
    for label in dist.samples():
        dString += "%s: %f " % (label, dist.prob(label))

    print( "True Features: ", dict_to_string( true_features(inFeatures, word_features_all) ) )

    # Classify using just a single, True feature.
    # Towards building stacked bar graph
    for w,v in true_features( inFeatures, word_features_all ).items():
        # single_f = find_features_with_debug( [w], w )
        single_f = find_features( [w], word_features_all )
        dist_f = classifier.prob_classify( single_f )

        fString = "%s -|-> (%s) " % (w, dict_to_string( true_features( single_f, word_features_all ) ) )
        fString += dist_f.max() + " : "
        for label in dist_f.samples():
            fString += "%s: %f " % (label, dist_f.prob(label))
        print( fString )

    return( predClass, dString )


def one_fold_classify(param_training_data):
    all_words = []

    for ws,c in param_training_data:
        for w in ws:
            all_words.append(w.lower())
    
    all_words = nltk.FreqDist(all_words)
    if verbose and len(all_words) < 50:
        print(  "All words: ", all_words )


    # All features words, used for classifier
    word_features_all = list(all_words.keys())[:num_features]
    if( num_features < len( list(all_words.keys()) ) ):
        print(  "*** Warning:  Only a subset of words used as features" )

    
    featuresets = [(find_features(rev, word_features_all), category) for (rev, category) in param_training_data]
    if verbose and len(featuresets) < 50:
        print(  "Feature Sets: ", featuresets )


    # Use fraction fold_split_percent to train, and rest to test
    fold_split =  int( round( len(featuresets) * fold_split_percent ) )

    # set that we'll train our classifier with
    training_set = featuresets[:fold_split]
    print( "Training Set Length: " + str(len(training_set)) + " of " + str( len(param_training_data)) )

    if verbose and fold_split < 5:
        print(  "Training Set Itself: ", training_set )

    # set that we'll test against.
    testing_set = featuresets[fold_split:]
    print( "Testing Set Length: " + str(len(testing_set)) + " of " + str( len(param_training_data)) )
    if verbose and len(testing_set) < 10:
        print(  "Testing Set Itself: ", testing_set )
    
    # Training/Testing set split
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    if verbose:
        print( dict_to_string_nl( classifier._feature_probdist ) )

    # Repeat this info, from above
    print( "Training Set Length: " + str(len(training_set)) + " of " + str( len(param_training_data)) )
    print( "Testing Set Length: " + str(len(testing_set)) + " of " + str( len(param_training_data)) )
    fold_accuracy = (nltk.classify.accuracy(classifier, testing_set))*100
    print("Classifier accuracy percent (training/testing):", fold_accuracy)

    classifier.show_most_informative_features(15)

    return( word_features_all, featuresets, fold_accuracy )
# end def



# ********************************************************

# TODO:  Should use command-line argument; should catch exceptions
comment_lines = 0
f = open("training.data", "r")
for line in f:
    l = line.strip()
    if len(l) <= 0:	# Skip blank lines
        continue
    if l[0] == "#":	# Skip comment lines
        if comment_lines < max_print_comment_lines:
            print(l)
        comment_lines += 1
        continue

    # Get labels
    if l[-1] == ":":	# Look for new classes/labels
        current_class = l[:-1]
        if verbose:
            print( "New class: " + current_class )
        continue
    elif l[-1] == "#":	# Look for new ngrams/features
        current_class = l[:-1]
        if verbose:
            print( "New ngrams/features: " + current_class )
        continue

    # Get data
    if current_class == "ngrams":
        ngrams.append( l )
    else:
        features = [l] + ([i for i in l.split()])
        fulltraining.append( ( features, current_class ) )
        fullsentence.append( (l, current_class) )
f.close()

print( "Comment lines: ", comment_lines )
print( "Ngrams: ", ngrams )

training_data = fulltraining
if verbose:
    print( "training_data: ", training_data )

# sys.exit(0)

min_a = 101
max_a = -1
for folds in range(0, 20):
    random.shuffle(training_data)
    Word_features_all, Featuresets, fold_accuracy = one_fold_classify(training_data)


## TODO:  Should reset/redo Word_features_all, etc. for all training data
training_set = Featuresets
testing_set = Featuresets
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent (training==testing):",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)

 
# print( Word_features_all )


# print ("************* (training==testing)) ***************" )


contFlag = True
while contFlag:
    inSentence = raw_input( "Test Sentence: " )
    if inSentence == "q":
        contFlag = False
    inWords = list( nltk.word_tokenize( inSentence ) )
    inWords.append( inSentence )
    print( "FINAL Classify: ", classify_tuple_with_debug( inWords, Word_features_all ) )
sys.exit( 0 )
