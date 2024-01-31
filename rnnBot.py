"""
This script builds a RNN for text generation. it can be trained or take an existing RNN.keras model.
When training, both a .keras model and a tokenizer.json are generated.
When passing in a RNN.keras model, a tokenizer must also be passed in. Both the model and the tokenizer should have the same name.
The model should be a .keras file, and the tokenizer a .json file.
There are septare functions for taking in uttances and generating a responses.
"""

import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import keras.layers
import json

class rnnBot:
    def __init__(self, modelName:str=""): 
        self.name = modelName
        self.tokenizer = Tokenizer()
        self.model = None
        try: 
            self.model = load_model(f"{modelName}.keras")
            with open(f"{modelName}.json", "r") as tokenizerJsonFile:
                self.tokenizer = tokenizer_from_json(tokenizerJsonFile.read())
        except:
            print(f"model:{modelName} does not exist")
        self.chatHistory = []
        self.chatHistoryLength = 0 #this is the number of tokens curently in the chat history
        
    def fit(self, texts:[str], newModelName:str) -> None: #only needs to be used if model has not been built yet
        """
        This is the training funrction for the RNN text generator. texts is a list of sentances in order, i.e. a conversation.
        New mdoel name is is the name given to the new model generated. The mdoel is saved as a .keras file
        """
        print("tokenization")
        # Tokenize the text data
        self.tokenizer.fit_on_texts(texts)
        total_words = len(self.tokenizer.word_index) + 1
        #Create input sequences and labels
        input_sequences = []
        for line in texts:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

        max_sequence_length = max(len(seq) for seq in input_sequences)
        padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

        # Create input and output sequences
        X = padded_sequences[:, :-1]
        y = padded_sequences[:, -1]

        # Convert y to one-hot encoded format
        y = keras.utils.to_categorical(y, num_classes=total_words)

        print("set up model")
        # Build the RNN model
        model = keras.Sequential([
            keras.layers.Embedding(input_dim=total_words, output_dim=50, input_length=max_sequence_length-1),
            keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True)),
            keras.layers.LSTM(units=100),
            keras.layers.Dense(units=total_words, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("train model")
        model.fit(X, y, epochs=100, verbose=1)# Train the model
        #save model and tokenizer
        model.save(newModelName + ".keras")#save model.
        tokenizer_json = self.tokenizer.to_json()
        with open(newModelName + ".json", "+w") as tokenizerJsonFile:
            tokenizerJsonFile.write(tokenizer_json)
        
        self.model = model #automaticaly set active model to new model generated

    def readInUtterance(self, utterance:str):
        """
        The language model uses a colection of utterances in sequance to determine/generate a response. by using a chat history, context is kept in the conversation.
        The lm struggles to process any more then 500 tokens at a time so we must keep track of the number. This is a rough count as tokenization chnages the number words to tokens.
        if the number of tokens (chatHistotyLength) exceeds 500, we remove the first utterance of the chat history as it is unlikely it is still relevent. 
        """
        self.chatHistoryLength += len(utterance.strip().split())
        if self.chatHistoryLength >= 500: 
            self.chatHistory = self.chatHistory[1:]
        self.chatHistory.append(utterance)
    
    def generateResponse(self) -> str:
        # Generate text using the trained model
        seed_text = "".join(self.chatHistory)
        #print(f"seed: {seed_text}")
        next_words = 15
        response = seed_text
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([response])[0]
            token_list = pad_sequences([token_list], maxlen=130, padding='pre')
            predicted_probs = self.model.predict(token_list, verbose=0)[0]
            predicted_index = np.argmax(predicted_probs)
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    
                    output_word = word
                    break
            #seed_text += " " + output_word
            response += " " + output_word
        return response.replace(seed_text, "") #remove the chat history for the response of the rnn bot
