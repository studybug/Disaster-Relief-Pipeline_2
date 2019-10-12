import sys

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    task: loads data put in,
       then converts pandas df
    return: X, y dataset
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    
    df = pd.read_sql_table('DisasterResponse', con=engine)

    X = df['message']
    y = df.drop(['id','message', 'original', 'genre'], axis=1)

    category_names = list(y.columns)
    #return df
    return X, y, category_names
   
                          

def tokenize(text):
    '''task: clean text by removing uppercase, puncuation,
    stopwords, tokenize, and then uses lemmatization
    return: tokens: tokens and lemmatization generated from the text'''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex , text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # load stop words
    stop_words = stopwords.words("english")
    
    # lemmatize andremove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    return tokens
    
    '''url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)  # find urls
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')  # replace urls

    tokens = word_tokenize(
        text)  # tokenizer object, not capitalised as it is a class method

    words = [word for word in tokens if word not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()  # parent class lemmatizer object

    clean_words = []  # empty list for results

    for word in words:
        clean_word = lemmatizer.lemmatize(
            word).lower().strip()  # return lemmatized words

        clean_words.append(clean_word)  # append cleaned/lemmatized string

    return clean_words
    
    
    #return tokens'''


def build_model():
    '''task: a pipeline model with CountVectorizer, TfidfTransformer, 
    MultiOutputClassifier with RandomForestClassifier - further parameters are added
    return: GridSearchCV pipeline'''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
     ])

    
    return pipeline

    parameters = {'vect__max_features': (None, 5000), 
              'clf__estimator__n_estimators': [10, 20] 
             } 
    
    model = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=1, n_jobs=-1)
    
    return model

def evaluate_model(model, X_test, y_test, category_names):
    '''input: testing model, ML tests, and names of categories
    return: Classification report featuring precision, recall, f1-score and count'''
     
    
    y_pred = model.predict(X_test)
    
    class_report = classification_report(y_test, y_pred, target_names=category_names)
    print(class_report)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
        print('Building model...')
        models = build_model()
        print(y.shape)
        print(X.shape)
        print('Training model...')
        models.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(models, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(models, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()