import speech_recognition
import pyttsx3
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from recommender_system import recommend, recommend_categories
import gensim.downloader as api
import json
import operator
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

nltk.download('stopwords')

# recognizer = speech_recognition.Recognizer()

# Stop words are the common high-frequency words that are not useful for analysis
stop_words = set(stopwords.words('english'))

#POS_TAGS are the type of tokens
#We have to pick the types which are useful for our analysis
impPostags = ['CD', 'NN', 'NNS', 'LS', 'NNP', 'VBG']
impWords = []

print("Importing the word vectors")
model = api.load('glove-wiki-gigaword-50')
word_vectors = model
print("Imported the word vectors")

def tokenizeFunc(text, data):
    positivity = True
    polarity = TextBlob(text).sentiment.polarity
    print("Polarity:", polarity)
    if float(polarity) < 0.0:
        positivity = False
    
    print("Inside tokenizeFunc")
    print(text)
    tokens = word_tokenize(text)
    
    # Removing Stop Words
    filtered_tokens = [word for word in tokens if word.casefold() not in stop_words]
    
    #POS Tagging
    posTags = pos_tag(filtered_tokens)
    print("POS Tags:")
    print(posTags)
    impWords = [item[0] for item in posTags if item[1] in impPostags]
    
    print("The important words are:")
    print(set(impWords))
    
    indices = dict()
    
    for word in set(impWords):
        recommended_catergories_indices = recommend_categories(word=word, data=data, word_vectors=word_vectors, positivity=positivity)
        for key, value in recommended_catergories_indices.items():
            # print(key,": ", value)
            try:
                if key not in indices:
                    indices[key] = value
                    # print(indices)
                else:
                    if positivity == False:
                        indices[key] = min(indices[key], value)
                    else:
                        indices[key] = max(indices[key], value)
            except:
                continue
    
    for word in set(impWords):
        recommended_indices = recommend(word=word, data=data, word_vectors=word_vectors, positivity=positivity)
        for key, value in recommended_indices.items():
            try:
                if key not in indices:
                    indices[key] = value
                else:
                    if positivity == False:
                        indices[key] = min(indices[key], value)
                    else:
                        indices[key] = max(indices[key], value)
            except:
                continue
    # print("Indices values after recommendation")
    # print(indices)
    
    indices = dict(sorted(indices.items(), key=operator.itemgetter(1), reverse=False if positivity == False else True))
    indices = {key: indices[key] for key in list(indices)[:3]}
    print("The final indices dict is as follows")
    print(indices)
        
    return [list(indices.keys()), list(impWords)]

@app.route('/', methods = ['GET'])
@cross_origin()
def speechRecog():
    recognizer = speech_recognition.Recognizer()
    print("Welcome to speech recognition system")
    print("Please pause for a while to let the system process your words")
    print("To quit, just speak 'exit'")
    text = ""
    # indices = dict()
    # while text != "exit":
    try:
        recData = request.args.get('recData')
        data = json.loads(recData)
        # print(recData)
        with speech_recognition.Microphone() as mic:
            recognizer.adjust_for_ambient_noise (mic, duration=0.2) 
            audio = recognizer.listen(mic)
            text = recognizer.recognize_google (audio)
            text = text.lower()
            return tokenizeFunc(text, data=data)
        
    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()
        return {}
    
if __name__ == "__main__":
    app.run()
