# Description       -   Self Learning chat-bot program
# Developer         -   Nicolas Euliarte Veliez
# Date of Creation  -   27/2/2020

# ----------------------------------------------------------------------------------------------------------------------
# Importing packages
# Article scraping tool = newspaper3k
# natural language toolkit = nltk

from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import warnings

# ----------------------------------------------------------------------------------------------------------------------
# Ignoring any warning messages
warnings.filterwarnings('ignore')

# Installing the nltk packages needed
# nltk.download('punkt')
# nltk.download('wordnet')

# ----------------------------------------------------------------------------------------------------------------------
# Getting the article URL
# https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521
webArticle = Article('https://towardsdatascience.com/top-algorithms-and-data-structures-you-really-need-to-know'
                     '-ab9a2a91c7b5')
webArticle.download()
webArticle.parse()
webArticle.nlp()

# collection of written text
corpus = webArticle.text

# print(corpus)

# ----------------------------------------------------------------------------------------------------------------------
# Tokenization
text = corpus
# converts text to list of sentences
sentenceTokens = nltk.sent_tokenize(text)

# print(sentenceTokens)

# ----------------------------------------------------------------------------------------------------------------------
# Creating dictionary removing punctuation
removePunctDictionary = dict((ord(punct), None) for punct in string.punctuation)


# Punctuation in text
# print(string.punctuation)
# Prints the pairs back with values
# print(removePunctDictionary)

# ----------------------------------------------------------------------------------------------------------------------
# Function returning limited lowercase words
def LimNorm(text):
    return nltk.word_tokenize(text.lower().translate(removePunctDictionary))


# Printing the list of word tokens
# print(LimNorm(text))

# ----------------------------------------------------------------------------------------------------------------------
# Key word matching

# Greeting inputs
greetInpt = ['hi', 'hello', 'hola', 'hey', 'greetings']

# Response
greetOutpt = ['hello', 'hi', 'hola', 'what\'s up']


# Returning random response to user
def Greeting(sentence):
    # Return greeting randomly if the user says a greeting
    for word in sentence.split():
        if word.lower() in greetInpt:
            return random.choice(greetOutpt)


# ----------------------------------------------------------------------------------------------------------------------
# ML aspect starting here
def Response(user_resp):
    # Make lowercase and ------ print
    user_resp = user_resp.lower()
    # print(user_resp)

    # Make empty string for chat-bot
    hugo_resp = ''

    # Append the user response to the list
    sentenceTokens.append(user_resp)

    # ------Print Sentence list after adding user response
    # print(sentenceTokens)

    # TFIDFVector object creation - term frequency and inverse document frequency multiplied together
    TfidfVec = TfidfVectorizer(tokenizer=LimNorm, stop_words='english')

    # Converting text to matrix of TF-IDF features
    tfidf = TfidfVec.fit_transform(sentenceTokens)

    # ------Print the TF-IDF features
    # print(tfidf)

    # Measure the similarity scores
    vals = cosine_similarity(tfidf[-1], tfidf)

    # Print them between 0 and 1
    # print(vals)

    # Get index of similar text/ sentence to user response
    idx = vals.argsort()[0][-2]

    # Reduce the vals - making it one list
    flattening = vals.flatten()

    # sort in ascending order
    flattening.sort()

    # get most similar to user response
    score = flattening[-2]

    # Printing similarity score
    # print(score)

    # If score  == 0 then the most similar score to user's response is no text similar to the users response
    if (score == 0):
        hugo_resp = hugo_resp + ' Sorry I dont understand'
    else:
        hugo_resp = hugo_resp + sentenceTokens[idx]

    # Print response
    # print(hugo_resp)

    # Removing the user response
    sentenceTokens.remove(user_resp)

    return hugo_resp


# ----------------------------------------------------------------------------------------------------------------------
flag = True
print("Ary: I am Ary, I will answer you questions about the given URL currently inside of me.")
while (flag == True):
    responseUser = input()
    responseUser = responseUser.lower()
    if (responseUser != 'bye'):
        if (responseUser == 'thanks' or responseUser == 'thank you'):
            flag = False
            print('Ary: You are welcome!')
        else:
            if(Greeting(responseUser) != None):
                print('Ary: '+Greeting(responseUser))
            else:
                print('Ary: '+Response(responseUser))

    else:
        flag = False
        print('Ary: See You Soon!')