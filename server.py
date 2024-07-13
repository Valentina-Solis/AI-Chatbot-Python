import numpy as np
import nltk
import string
import random

#Reading Text Corpus - Stage 1
data = open('D:\Chatbox\info.txt', 'r', errors = 'ignore')
raw_data = data.read()

#print(raw_data)

raw_data = raw_data.lower()
#nltk.download('punkt') #Use of the tokenizer
#nltk.download('wordnet') #Use of the dictionary
#nltk.download('omw-1.4')

#print(raw_data)

tokenized_sentence = nltk.sent_tokenize(raw_data)
tokenized_words = nltk.word_tokenize(raw_data)

#print(tokenized_sentence[:2])
#print(tokenized_words[:2])

#Process the text - Stage 2
lemmas = nltk.stem.WordNetLemmatizer()

def LemmaTokens(tokens):
    return [lemmas.lemmatize(token) for token in tokens]

remove_punctuation = dict((ord(punct), None) for punct in string.punctuation)

def LemmaNormalize(text):
    return LemmaTokens(nltk.word_tokenize(text.lower().translate(remove_punctuation)))

#Function to greet
greet_inputed = ('hello', 'hi', 'what is up', 'how are you going?')
greet_responses = ('Hi', 'Hey there', 'Hello', 'Hey!')

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputed:
            return random.choice(greet_responses)
        
#Generation of answer by the bot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def answer(user_response):
    bot_response = ''
    Vector = TfidfVectorizer(tokenizer = LemmaNormalize, stop_words='english')
    tfidf = Vector.fit_transform(tokenized_sentence) #Transforms the sentence into a TF-IDF vector
    values = cosine_similarity(tfidf[-1], tfidf)
    index = values.argsort()[0][-2] #Find the most similar, ascending order 
    flat = values.flatten() #Multi dimensional array to 1D array
    flat.sort() #increasing order
    required_tfidf = flat[-2]
    if(required_tfidf == 0):
        bot_response = bot_response + "Sorry, I am unable to understand you"
        return bot_response
    else:
        bot_response = bot_response + tokenized_sentence[index]
        return bot_response
    
flag = True
print('Hey, I am an AI leargning Bot. Start typing after greeting to talk to me and for ending conversation type bye.')
while(flag == True):
    user_answer = input()
    user_answer = user_answer.lower()
    if(user_answer != 'bye'):
        if(user_answer == 'thank you' or user_answer == 'thanks'):
            flag = False
            print('Bot: You are welcome!!')
        else:
            if(greeting(user_answer) != None):
                print('Bot: ' + greeting(user_answer))
            else:
                tokenized_sentence.append(user_answer)
                tokenized_words = tokenized_words + nltk.word_tokenize(user_answer)
                final_word_set = list(set(tokenized_words))
                print('Bot: ', end ='')
                print(answer(user_answer))
                tokenized_sentence.remove(user_answer)
    else:
        flag = False
        print('Bot: Goodbye')
    