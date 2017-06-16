# NLIzer

##Summary
A Neural Network for Native Language Inference written in Tensorflow 

##What it does
Native language inference is an attempt to determine what a person's native language is based on writing or speech samples in a second language. For example, if a person grew up speaking German and learned Spanish later in life, we would try determine that they grew up speaking German based on a writing sample they wrote in Spanish. Below I have included a confusion matrix for a different but simmilar project. In a perfectly working system, the graph would be completely blue along the diagonal and white everywhere else. 

![](http://imgur.com/a/EDi30)

##How it works 
The data I used is from the [Instituto Cervantes](http://www.cervantes.es/default.htm), a non-profit organization that seeks to promote the study and teaching of Spanish worldwide. The data set is from the Corpus de Aprendices de Español, or "Corpus of Spanish Learners", people who have studied Spanish for different amounts of time and have taken a standarized test. In the test people are given different topics to work on and must produce essays, which are then graded for different levels of fluency. We do not have access to the exact essays that they wrote, but instead we have an annotated version of their essays that include different language features. More information about what the annotations mean can be found [here](). I then changed all the annotations into numbers so VB (which is a verb) may be encoded as a one. Then a sentence is changed into a vector that can be analyzed by the LSTM Neural Network that I wrote in Tensorflow. 

##Results

##Research and more info 
