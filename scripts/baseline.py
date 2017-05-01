from csv import DictReader
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
#read the data from csv file
def read_data(name):
    text, targets = [], []

    #data has 2 column labels text which is the whisper and category which is the label
    with open('data/{}.csv'.format(name)) as f:
        for item in DictReader(f):
            text.append(item['text'].decode('utf8'))
            targets.append(item['category'])
    return text, targets
    #this function returns separated text and its label

def main():
    text_train, targets_train = read_data('train')
    text_test, targets_test = read_data('test')

    #create a model using pipeline,tfid classifier and Random forest classifier
    model = make_pipeline(
        TfidfVectorizer(),
        RandomForestClassifier(),
    ).fit(text_train, targets_train)

    #predict using the model
    prediction = model.predict(text_test)

    
    print 'macro f1:', f1_score(targets_test, prediction, average='weighted')

if __name__ == "__main__":
    main()
