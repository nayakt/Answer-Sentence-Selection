How to run!!!!!
---------------------------

1. Copy Ans-Sent-Selection-Proj folder in some location.
2. Download Google's pre-trained word vectors from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit in Ans-Sent-Selection-Proj/data folder
3. Extraxt the binary file in same location and it should be named as 'GoogleNews-vectors-negative300.bin'
4. Open Ubuntu terminal and move to Ans-Sent-Selection-Proj/code folder.
5. Use following command to run a model:
	a) Bag of Words Model:
		python CNNQAClassifier.py BoW GoogleNews-vectors-negative300.bin stopwords.txt WikiQASent-train.txt WikiQASent-dev-filtered.txt WikiQASent-dev-filtered.ref WikiQASent-test-filtered.txt WikiQASent-test-filtered.ref
	b) Bigram CNN Model:
		python CNNQAClassifier.py BigramCNN GoogleNews-vectors-negative300.bin stopwords.txt WikiQASent-train.txt WikiQASent-dev-filtered.txt WikiQASent-dev-filtered.ref WikiQASent-test-filtered.txt WikiQASent-test-filtered.ref
	c) Trigram CNN Model:
		python CNNQAClassifier.py TrigramCNN GoogleNews-vectors-negative300.bin stopwords.txt WikiQASent-train.txt WikiQASent-dev-filtered.txt WikiQASent-dev-filtered.ref WikiQASent-test-filtered.txt WikiQASent-test-filtered.ref
	d) Decomposition and Composition based CNN Model:
		python CNNQAClassifier.py DecompCompCNN GoogleNews-vectors-negative300.bin stopwords.txt WikiQASent-train.txt WikiQASent-dev-filtered.txt WikiQASent-dev-filtered.ref WikiQASent-test-filtered.txt WikiQASent-test-filtered.ref

