# EEG-Classification

by Weinan Song, Tianxue Chen, Ying Wang

## Introduction
Electroencephalography (EEG) is a noninvasive method to measure human brain singles, which embedded rich information
of human thoughts. And the object of this work is to classify human being’s imagination activities based on EEG raw data.

Deep learning is an artificial approach of training the input data with parameters to lead a better classification or prediction on new data. Current popular networks are CNN and RNN under various conditions. As CNN using connectivity pattern between its neurons and RNN using timeseries information, we proposed both networks and combine these two networks to do the classification and compared the performance.

Convolution neural networks(CNN) has shown great performance in classifying images since [2]. Many problem in computer vision can be efficiently solved based on CNN. In this paper, we treat the sequence data as images and change the problem into a basic image classification problem.

Recurrent neural networks(RNN) is an efficient algorithm which can be used in dealing with sequence data. The algorithm has proved it performance in processing natural language processing(NLP) like [4]. In this paper, we also change this problem into a NLP problem by taking each 22x1 vector as a word vector and the whole sequence as a word.
