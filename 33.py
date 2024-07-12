#1 tokenizing

import nltk 
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text="NLP is quite fascinating"
tokens=word_tokenize(text)
print(tokens)


#2stemming

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words=['eating','eats','ate']
wordss=["running","ran","runs"]
stems=[stemmer.stem(word) for word in words]
print(stems)


#3 Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
words=['eating','eats','ate']
lemmas=[lemmatizer.lemmatize(word,pos='v') for word in words]
print(lemmas)

#4 stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
#print(stop_words)
filtered_text= [word for word in tokens if word.lower() not in stop_words]
print(filtered_text)
