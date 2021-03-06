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
* CNN
  * Download the preprocesseddata from Google drive: https://drive.google.com/open?id=10Ix-i7orrBuVIKLQGSc5x1UFJ4WdbW48
  * Run the CNN notebook one cell by one cell.
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

## Installation
* NLTK and wordnet  
Install NLTK
	```
    	pip install nltk
	```  
	Download Wordnet through NLTK in python console:
	```
	    import nltk
	    nltk.download('wordnet')
	    nltk.download('averaged_perceptron_tagger')
	    nltk.download('punkt')
	    nltk.download('word2vec_sample')
	```
	if you have problem running the code because of lacking nltk package, following the instructions in the output.

* Gensim
```
pip install --upgrade gensim
```
or
```
conda install -c conda-forge gensim
```
* Paramiko
```
pip install paramiko
```

## Preprocessing 
### Training
* Run pre_train.py with training file (contains human annotated amr) e.g. amr-bank-struct-v1.6-training.txt
	```
	    python pre_train.py amr-bank-struct-v1.6-training.txt
	```	
	This will output two files: for_pos.txt, amr-bank-struct-v1.6-training.csv

* If you want to us our CNN model, open preprocessing_for_nn.ipynb and change input file name to the the csv file produced from previous step. And change the output file name to the name you want. Change mode to "five" to get input file for cnn model. 
	* This will output two file: isfocus_divby5_fname.npy, word_embedding_divby5_fname.npy  
		

### Parsing
* run SVM+LR.py with Spyder and run afterwards post-processing.py
* run learn_word_embedding_and_keras.ipynb with Jupyter Notebook
* run CNN_5s.ipynb with Jupyter Notebook (note: the example input file is too big to upload to github. you need to run preprocessing_for_cnn.ipynb and changed )


## Focus Identification

## Paper on the state-of-the-art
* https://arxiv.org/pdf/1805.05286.pdf
* Uses Bi-LSTMs, ,lets maybe use a predefined library
>>>>>>> origin/master
