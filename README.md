# QuoraInsincereQuestions
Kaggle competition to identify insincere questions on Quora. In addition to RF, SVM, LSTM, and biLSTM attention 
network attempts, a custom network, "LDA-Attention" is used, in which biLSTM Attention sub-networks are trained
concurrently, one for each topic obtained from Latent Dirichlet Allocation. The final layer is a dot product of 
the sub-network outputs with the topic weights of the sentence (so, the topic weights are serving as sub-network 
weights). The motivation for this network was large observed differences in topic weight distributions between 
insincere and sincere questions. Given the subtlety often displayed in insincere questions, the idea was to train 
the weights to be less broad and more topic-specific. 

Description of the files...

1) quorasignificanttokens.py : Investigates which words are most indicative of both insincere and sincere questions.
2) quorabasicmodel.py : Applies basic RF/SVM models to the dataset, with either tf-idf or BoW as features. Looks
    at what kinds of sentences are most easily identified correctly vs. incorrectly.
3) quorann.py : Applies both a standard LSTM model as well as a biLSTM with Attention model. Cyclic learning rates.
4) quoraldaandemb.py : Performs Latent Dirichlet Allocation on the dataset, runs the LDA-Attention model.
5) quorachecklist.txt : Includes information pertaining to the dataset as well as results obtained from various 
    attempts.
    
To access the dataset, one must have accepted the rules of the competition on Quora. 
