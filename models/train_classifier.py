import re
import sys
import nltk
import pickle
import pandas as pd

from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def load_data(database_filepath):
    """
    Load Data from the Database

    Arguments:
        database_filepath -> Path to SQLite destination database
    Output:
        X -> Dataframe containing features
        y -> Dataframe containing labels
        category_names -> List of categories name
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', con = engine)

    # since the values are all 0 in child_alone, we drop this colomn
    df = df.drop('child_alone', axis=1)

    # I always get an Error at first, after asked this ValueError in Knowledge, i set the related = 2 to related = 1
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

    X = df['message']
    y = df.iloc[:,4:]

    category_names = y.columns
    return X, y, category_names


def tokenize(text, url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text

    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    # The first step, i replace the url in message to avoid its influence 
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # The second step is to extract the words from step, we could use word_tokenize
    tokens = nltk.word_tokenize(text)

    # The third step, lemmatizer to get the original form of the words and clean them
    lemmatizer = nltk.WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """
    # this part i take from ML Learning part,  
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build Pipeline

    Output: 
        the model (Pipeline) that process text messages and apply a classifier
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # for this setting, i only do the GridSearchCV very clear 
    # because it will overload the limitation of Github, if GridSearchCV works a lot
    # i want to upload the code in grey, they can work, but when i upload them on Github
    # remote: error: File models/classifier.pkl is 232.90 MB; this exceeds GitHub's file size limit of 100.00 MB
    # remote: error: File models/classifier.pkl is 262.43 MB; this exceeds GitHub's file size limit of 100.00 MB

    #param_grid = {
        #  'clf__estimator__n_estimators': [1, 101, 10],
        #  'clf__estimator__max_features': [1, 10, 1]
    #}
    #model = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=2, verbose=2, cv=2)

    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, y_pred, target_names=category_names)
    print(class_report)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
