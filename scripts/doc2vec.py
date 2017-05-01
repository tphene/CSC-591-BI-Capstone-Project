from csv import DictReader
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
import nltk
import random
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import sklearn.linear_model
import sklearn.naive_bayes
#read the data and break into text,target
def read_data(name):
    text, targets,text_sl = [], [], []

    with open('data/{}.csv'.format(name)) as f:
        for item in DictReader(f):
                targets.append(item['category'])
                text.append(item['text'])

    return text, targets

def list_text(text):
    
    # removing stop words from the whispers
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text_sl = []
    # sum = 0

    for line in text:
        words = [w.lower() for w in line.strip().split() if (w not in stopwords and len(w)>=3)]
        # sum = sum + len(words)
        text_sl.append(words)
    
    # average number of words per whisper to find optimal value for 'window' in doc2vec
    # print(float(sum)/len(text_sl)) 
    return text_sl


def feature_vecs_doc(text_sl,test_sl):
    # Obtaining feature vectors from the data
    #label the training and testing data with tags and numbers 

    #empty list that will contain the training data with tags
    labeled_train_text = []

    for i,whisperWords in enumerate(text_sl):
        lso=LabeledSentence(words=whisperWords,tags=['TRAIN_'+str(i)])
        labeled_train_text.append(lso)


    #empty list that will contain the testing data with tags
    labeled_test_text = []
    for i,whisperWords in enumerate(test_sl):
        lso=LabeledSentence(words=whisperWords,tags=['TEST_'+str(i)])
        labeled_test_text.append(lso)

    #return the training and testing data with tags
    return labeled_train_text,labeled_test_text

def make_vecs(labeled_train_text,labeled_test_text,train_sl,test_sl):

    #Create a model for doc2vec with window size 8 , minimum count 10 and size 100
    model = Doc2Vec(window = 8,min_count=10,size = 100) 
    #combine the training and testing data
    sentences = labeled_train_text + labeled_test_text
    #build vocab using the combined training and testing data
    model.build_vocab(sentences)
    for i in range(5):
        print "Training iteration", i
        #shuffle the sentences 
        random.shuffle(sentences)
        #train the model using shuffled sentences
        model.train(sentences)

    #initialize an empty vector training list
    train_vec=[]

    #fill the list using the vectors associated with that particular tag
    for i,fv in enumerate(train_sl):
        featureVec = model.docvecs['TRAIN_'+str(i)] 
        train_vec.append(featureVec)
    
    #initialize an empty vector training list
    test_vec=[]

    #fill the list using the vectors associated with that particular tag
    for i,fv in enumerate(test_sl):
        featureVec = model.docvecs['TEST_'+str(i)]
        test_vec.append(featureVec)

    #return the lists which are vectors associated with tags
    return train_vec,test_vec


def main():
    text_train, targets_train = read_data('train')
    text_test, targets_test = read_data('test')

    # Creating Lists of lists of words
    train_sl = list_text(text_train)
    test_sl = list_text(text_test)

    #call to the function creating document vectors which are features for classification
    labeled_train_text,labeled_test_text = feature_vecs_doc(train_sl,test_sl)

    train_vec,test_vec = make_vecs(labeled_train_text,labeled_test_text,train_sl,test_sl)
    X = train_vec
    Y = targets_train
    
    #logistic regression is used for classification
    lm  = sklearn.linear_model.LogisticRegression()
    #fit the logistic regression model
    lr_model = lm.fit(X,Y)

    #predict using the created model
    predictionLM = lr_model.predict(test_vec)
    #print the f1 score or f1 measure
    print 'macro f1 for linear model:', f1_score(targets_test, predictionLM, average='macro')
    # print 'weighted f1 for linear model:', f1_score(targets_test, predictionLM, average='weighted')

    nb  = sklearn.naive_bayes.BernoulliNB()
    #fit the BERNOULLI naive bayes model
    nb_model = nb.fit(X,Y)

    #predict using the created model
    predictionNB = nb_model.predict(test_vec)
    print 'macro f1 for Naive Bayes model:', f1_score(targets_test, predictionNB, average='macro')
    # print 'weighted f1 for Naive Bayes model:', f1_score(targets_test, predictionNB, average='weighted')



if __name__ == "__main__":
    main()
