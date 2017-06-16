#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import numpy as np
import tensorflow as tf
from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer

class RNN(object):
	def __init__(self, testnum):
		self.testnum = testnum


	#translates language file to english
	def TranslateLangFile(self):
		language = open('EnglishLangOut.txt', 'w')
		original = open("langout.txt", "r")
		for line in original:
			if ('Ruso' in line):
				language.write("Russian\n") 
			elif ('rabe' in line):
				language.write("Arabic\n")
			elif ('Port' in line):
				language.write("Portuguese\n")
			elif ('Franc' in line):
				language.write("French\n")
			elif ('Chino' in line):
				language.write("Chinese\n")
			elif('Ingl' in line):
				language.write("English\n")
			else:
				print("Language not caught: " + line)


	#creates outputs 
	def correctAnswer(self): #
		retVector = []
		languages = open("EnglishLangOut.txt", "r")
		baseLang = ['Chinese', 'Russian', 'English', 'Portuguese', 'Arabic' , 'French']
		for x in languages.readlines(): 
			x = x.strip('\n')				
			retVector.append(baseLang.index(x))
		return retVector

	#vectorizes the answer list to be in a 1 hot array
	def vectorizeCorrectAnswer(self, array):
		retVector = []
		for i in array:
			temp_list = np.zeros((6,), dtype=np.int)
			temp_list[i]=1
			retVector.append(temp_list)
		return retVector

	#creates an array of the tags with standard length of 945(the longest string in the set)
	def manualVectorize(self):
		file = open("tags.txt", "r")
		with open('CaesTags.txt') as f:
			comparitorlist = f.read().splitlines()
		returnArray = []
		for i in file.readlines():
			tagList = i.split()
			inner = []
			for j in tagList: 
				if j in comparitorlist:
					inner.append([comparitorlist.index(j)])
				else:
					print(j)
			if(len(inner) < 945):
				while(len(inner) < 945):
					inner.append([0])		
			returnArray.append(np.array(inner))
		return returnArray 	

			
	def run1DRNN(self, numberofExamples, batchSize, epochNum):
		#getting input and output data
		inputVector = self.manualVectorize()
		outputVector = self.vectorizeCorrectAnswer(self.correctAnswer())
		#setting up training and testing data
		NUM_EXAMPLES = numberofExamples
		train_input = inputVector[:NUM_EXAMPLES]
		train_output = outputVector[:NUM_EXAMPLES] #everything up until NUM_EXAMPLES
		test_input = inputVector[NUM_EXAMPLES:]
		test_output = outputVector[NUM_EXAMPLES:] #everything beyond NUM_EXAMPLES to the end
		#Setting up the RNN, tells what input and output should be expected
		data = tf.placeholder(tf.float32, [None, 945,1])
		target = tf.placeholder(tf.float32, [None, 6])
		#setting up the LSTM cell
		num_hidden = 24
		cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
		val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
		#manipulating the data
		val = tf.transpose(val, [1, 0, 2])
		last = tf.gather(val, int(val.get_shape()[0]) - 1)
		#setting up weights
		weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))#truncated normal is the type of random dist
		bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
		prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
		cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
		#setting up optimizer
		optimizer = tf.train.AdamOptimizer()#uses adam gradient descent, might be an area of experimentation 
		minimize = optimizer.minimize(cross_entropy)
		#setting up mistake and error calculators
		mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
		error = tf.reduce_mean(tf.cast(mistakes, tf.float32))#gets the mean of the errors
		init_op = tf.global_variables_initializer() #tf.initialize_all_variables
		#starting session
		sess = tf.Session()
		sess.run(init_op)
		batch_size = batchSize
		no_of_batches = int(len(train_input)/batch_size)
		epoch = epochNum
		print("Running test with number of examples " + str(numberofExamples) + " batch size of " + str(batchSize) + " and epoch number " + str(epochNum) + "\n")
		#note: One Epoch = one forward pass and one backward pass of all the training examples
		#batch size = number of training examples in one forward/backward pass
		#iterations = number of passes
		with open("SpanishTest3.txt", "a") as file:
			file.write("Test with number of examples " + str(numberofExamples) + " batch size of " + str(batchSize) + " and epoch number " + str(epochNum) + "\n")
		epochlist = 0
		for i in range(epoch):
			print(i)
			ptr = 0
			for j in range(no_of_batches):
				inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
				ptr+=batch_size
				sess.run(minimize,{data: inp, target: out})
			#error calculation, averaged out every 100 epochs
			epochlist += sess.run(error,{data: test_input, target: test_output})
			if (i % 100 == 0):
				if (i == 0):
					incorrect = sess.run(error,{data: test_input, target: test_output})
					print('Epoch {:2d} error {:3.1f}% \n'.format(i, 100 * incorrect))
					with open("FullSentenceTest3.txt", "a") as file:
						file.write('Epoch {:2d} error {:3.1f}% \n'.format(i, 100 * incorrect)) 
				else:
					incorrect = epochlist/100
					print('Epoch {:2d} error {:3.1f}% \n'.format(i, 100 * incorrect))
					with open("FullSentenceTest3.txt", "a") as file:
						file.write('Epoch {:2d} error {:3.1f}% \n'.format(i, 100 * incorrect)) 	
					epochlist = 0
		incorrect = sess.run(error,{data: test_input, target: test_output})
		print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
		sess.close()




testnumber1 = RNN(1)
testnumber1.run1DRNN(numberofExamples = 1900, batchSize = 1900, epochNum = 1000)
#print(testnumber1.createArray()[0])



