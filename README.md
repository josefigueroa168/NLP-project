# NLP-project

We plan to create a focus identification process through NLP paradigms, to assist the accuracy of AMR parsing.

## Implementation Ideas

* Hidden Markov Model?
  * Use POS tags and calculate how often words in the corpus are tagged as focus to predict focus of future sentences
  * Maybe add 'focus' to be its own POS tag..
* PCA
  * Again use POS tags and find features that are most responsible for a focus: 
    * Feature 1: Words tag
	* Feature 2/3: Tag of previous/next word
    * Feature 4/5: Tag of 2nd previous/next word
* Have tokenized sentences
* tensor flow (dnn)
### Libraries
* python: dnn
* NLTK
https://pypi.org/project/dnn/
* StandfordCoreNLP
https://nlp.stanford.edu/software/lex-parser.html


## Individual Tasks
* Code to parse datasets into neat python data structures(Dictionary, pandas Dataframe etc)
* Code algorithms to output probability
  * More specifically, define features to use in parsed datasets and feed to one of:
    * PCA
	* HMM
	* LSTM
* Code a testing suite to test model on new data and predict accuracy (Simple)

## Preprocessing
* writing shell file now....  

## Focus Identification