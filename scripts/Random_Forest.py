from csv import DictReader
from sklearn.linear_model import LogisticRegression
import sklearn.naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import nltk
from collections import Counter

def read_data(name):
    text, targets = [], []

    with open('data/{}.csv'.format(name)) as f:
        for item in DictReader(f):
            text.append(item['text'].decode('utf8'))
            targets.append(item['category'])

    return text, targets


def main():
    text_train, targets_train = read_data('train')
    text_test, targets_test = read_data('test')

    #Creating a model using the pipeline and a random forest classifier
    model = make_pipeline(
        TfidfVectorizer(strip_accents='unicode',stop_words='english',max_features=600,sublinear_tf=True),
       RandomForestClassifier(),
    ).fit(text_train, targets_train)
    #Fit the logistic regression model and predict the targets and store them in a list
    prediction = list(model.predict(text_test))  

    # Manually Engineered Features for Military,Fashion, Sports and Faith
    stopwords = set(nltk.corpus.stopwords.words('english'))
    for i,line in enumerate(text_test):
        words = [w.lower() for w in line.strip().split() if (w not in stopwords and len(w)>=3)]
        if('millitary' in words or 'veteran' in words or 'marine' in words or 'army' in words or 'military' in words):
            prediction[i] = 'military'
        if('baseball' in words or 'gym' in words or 'soccer' in words or 'football' in words):
            prediction[i] = 'sports'
        if('jesus' in words or 'bible' in words or 'god' in words or 'christian' in words):
            prediction[i] = 'faith'
        if('heels' in words or 'flats' in words or 'hair' in words or 'jeans' in words or 'dress' in words):
            prediction[i] = 'fashion'
        if('tattoo' in words or 'tattoos' in words):
            prediction[i] = 'tatoos'

    # Checking count of categories appearing in the prediction
    # counts = Counter(prediction)
    # print counts


    print 'macro f1:', f1_score(targets_test, prediction, average='macro')
    # print 'weighted f1:', f1_score(targets_test, prediction, average='weighted')

if __name__ == "__main__":
    main()
