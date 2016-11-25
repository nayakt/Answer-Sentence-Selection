# Answer Sentence Selection

Tested on Ubuntu 16.04 LTS with Python 2.7

How to run the code:

1. Copy `Ans-Sent-Selection-Proj` folder in some location. (optional)
2. Download Google's pre-trained word vectors from <https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit> in `Ans-Sent-Selection-Proj/data` folder
3. Unzip the downloaded file `GoogleNews-vectors-negative300.bin.gz` in the same location and rename it as `GoogleNews-vectors-negative300.bin`
4. Open console or terminal and change the current working directory to `Ans-Sent-Selection-Proj/code`
5. Use the specified command below (without newline) to run each type of model

 1. Bag of Words Model:  
    `python CNNQAClassifier.py BoW GoogleNews-vectors-negative300.bin stopwords.txt WikiQASent-train.txt WikiQASent-dev-filtered.txt WikiQASent-dev-filtered.ref WikiQASent-test-filtered.txt WikiQASent-test-filtered.ref`
 2. Bigram CNN Model:  
    `python CNNQAClassifier.py BigramCNN GoogleNews-vectors-negative300.bin stopwords.txt WikiQASent-train.txt WikiQASent-dev-filtered.txt WikiQASent-dev-filtered.ref WikiQASent-test-filtered.txt WikiQASent-test-filtered.ref`
 3. Trigram CNN Model:  
    `python CNNQAClassifier.py TrigramCNN GoogleNews-vectors-negative300.bin stopwords.txt WikiQASent-train.txt WikiQASent-dev-filtered.txt WikiQASent-dev-filtered.ref WikiQASent-test-filtered.txt WikiQASent-test-filtered.ref`
 4. Decomposition and Composition based CNN Model:  
    `python CNNQAClassifier.py DecompCompCNN GoogleNews-vectors-negative300.bin stopwords.txt WikiQASent-train.txt WikiQASent-dev-filtered.txt WikiQASent-dev-filtered.ref WikiQASent-test-filtered.txt WikiQASent-test-filtered.ref`

