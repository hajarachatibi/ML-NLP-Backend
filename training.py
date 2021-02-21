import pickle
import re
import string
import pandas as pd
import pymongo
from mongoengine import connect
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# --- Function to get collected data from database --- #
def getDataFromDB():
    try:
        # --- Connection to Mongodb --- #
        mongo = pymongo.MongoClient(
        host = "localhost",
        port = 27017,
        serverSelectionTimeoutMS = 1000,
        )
        db = mongo.ml_nlp_db

        # --- Connection to our database --- #
        connect('ml_nlp_db', host='mongodb://localhost', alias='default')

        # --- Getting data from data collection (table) --- #
        cursor = db['Data'].find()

        ids = []
        title = ' '
        content = ' '
        scores = []
        classes = []
        combined_title_content = []

        # --- Loop --- #
        for document in cursor:
            ids.append(document['_id'])
            title = document['Title']
            content = document['News']
            combine = title + " " + content
            combined_title_content.append(combine)
            scores.append(document['Score'])
            classes.append(document['Class'])
   
        # --- Put collected information into a Json form (keys: ids, values: combined_title_content) --- #
        content_by_id = {}
        for i, eid in enumerate(ids):
            content_by_id[eid] = combined_title_content[i]

        return {'content': content_by_id, 'score': scores, 'class': classes}

    except Exception as ex:
      print(ex)

# --- Put the list of combined_title_data from a list of text into a string format --- #
def putIntoString(listOfText):
    string_text = ''.join(listOfText)
    return string_text

# --- Function to put our collected data into a pandas DataFrame --- #
def putDataInDataFrame(string_text):
    data_df = pd.DataFrame.from_dict(string_text).transpose()
    data_df.columns = ['content']
    data_df = data_df.sort_index()
    return data_df

# --- Function to clean the data --- #
def cleanData(text):
    text = re.sub('\[.*?\]', '', text) # Remove everything between []
    text = re.sub('\(.*?\)', '', text) # Remove everything between ()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub('\w*\d\w*', '', text) # Remove numbers
    text = re.sub('[‘’“”«»…]', '', text) # Remove specific caracters
    text = re.sub('\n', '', text) # Remove '\n'
    return text

# --- Function to organize the Dataframe --- #
def organizeData(data_df, scores, classes, cleaned_data):
    # --- Adding score and class columns --- #
    data_df['score'] = scores
    data_df['class'] = classes
    return data_df

# --- Training models class (KNN, DT, ANN, NB, SVM) --- #
class TrainingModels:
    organized_data = ''

    # --- Training model function --- #
    #----------------SVM---------------
    def svmtrain(organized_data):
        features = ['content', 'score', 'class']
        organized_data = organized_data.drop_duplicates() # Remove duplicated rows
        organized_data.reset_index(drop=True, inplace=True) # Remove dataframe indexes
        Y = organized_data['class'].values # We fixe the column class in Y
        X = organized_data.drop(columns=['class']) # We drop it in X

        X_train, X_test, Y_train, Y_test = train_test_split(organized_data['content'], Y, test_size=0.3)

        pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', SVC() )]) # SVM

        # Fitting the model
        model = pipe.fit(X_train, Y_train)

        # Saving model
        pickle.dump(model, open('modelsvm.sav', 'wb'))

        # Prediction of X_test
        print(Y_test)
        prediction = model.predict(X_test)
        print(prediction)

        # Evaluation
        precision, recall, fscore, support = score(Y_test, prediction)
        print("------------Evaluation-----------------")
        # Accuracy
        print("SVM accuracy: {}%".format(round(accuracy_score(Y_test, prediction) * 100, 2)))
        # precision
        print('precision: {}'.format(precision))
        # recall
        print('recall: {}'.format(recall))
        # fscore
        print('fscore: {}'.format(fscore))
        # support
        print('support: {}'.format(support))

        return model

    #----------------KNN-----------------
    def knntrain(organized_data):
        features = ['content', 'score', 'class']
        organized_data = organized_data.drop_duplicates() # Remove duplicated rows
        organized_data.reset_index(drop=True, inplace=True) # Remove dataframe indexes
        Y = organized_data['class'].values # We fixe the column class in Y
        X = organized_data.drop(columns=['class']) # We drop it in X

        X_train, X_test, Y_train, Y_test = train_test_split(organized_data['content'], Y, test_size=0.3)

        pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', KNeighborsClassifier() )]) # KNN

        # Fitting the model
        model = pipe.fit(X_train, Y_train)

        # Saving model
        pickle.dump(model, open('modelknn.sav', 'wb'))

        # Prediction of X_test
        print(Y_test)
        prediction = model.predict(X_test)
        print(prediction)

        # Evaluation
        precision, recall, fscore, support = score(Y_test, prediction)
        print("------------Evaluation-----------------")
        # Accuracy
        print("KNN accuracy: {}%".format(round(accuracy_score(Y_test, prediction) * 100, 2)))
        # precision
        print('precision: {}'.format(precision))
        # recall
        print('recall: {}'.format(recall))
        # fscore
        print('fscore: {}'.format(fscore))
        # support
        print('support: {}'.format(support))

        return model

    #--------------ANN-----------------

    def anntrain(organized_data):
        features = ['content', 'score', 'class']
        organized_data = organized_data.drop_duplicates() # Remove duplicated rows
        organized_data.reset_index(drop=True, inplace=True) # Remove dataframe indexes
        Y = organized_data['class'].values # We fixe the column class in Y
        X = organized_data.drop(columns=['class']) # We drop it in X

        X_train, X_test, Y_train, Y_test = train_test_split(organized_data['content'], Y, test_size=0.3)

        pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', MLPClassifier() )]) # ANN

        # Fitting the model
        model = pipe.fit(X_train, Y_train)

        # Saving model
        pickle.dump(model, open('modelann.sav', 'wb'))

        # Prediction of X_test
        print(Y_test)
        prediction = model.predict(X_test)
        print(prediction)

        # Evaluation
        precision, recall, fscore, support = score(Y_test, prediction)
        print("------------Evaluation-----------------")
        # Accuracy
        print("ANN accuracy: {}%".format(round(accuracy_score(Y_test, prediction) * 100, 2)))
        # precision
        print('precision: {}'.format(precision))
        # recall
        print('recall: {}'.format(recall))
        # fscore
        print('fscore: {}'.format(fscore))
        # support
        print('support: {}'.format(support))

        return model

    #------------DT----------------

    def dttrain(organized_data):
        features = ['content', 'score', 'class']
        organized_data = organized_data.drop_duplicates()  # Remove duplicated rows
        organized_data.reset_index(drop=True, inplace=True)  # Remove dataframe indexes
        Y = organized_data['class'].values  # We fixe the column class in Y
        X = organized_data.drop(columns=['class'])  # We drop it in X

        X_train, X_test, Y_train, Y_test = train_test_split(organized_data['content'], Y, test_size=0.3)

        pipe = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('model', DecisionTreeClassifier())])  # DecisionTree

        # Fitting the model
        model = pipe.fit(X_train, Y_train)

        # Saving model
        pickle.dump(model, open('modeldt.sav', 'wb'))

        # Prediction of X_test
        print(Y_test)
        prediction = model.predict(X_test)
        print(prediction)

        # Evaluation
        precision, recall, fscore, support = score(Y_test, prediction)
        print("------------Evaluation-----------------")
        # Accuracy
        print("DT accuracy: {}%".format(round(accuracy_score(Y_test, prediction) * 100, 2)))
        # precision
        print('precision: {}'.format(precision))
        # recall
        print('recall: {}'.format(recall))
        # fscore
        print('fscore: {}'.format(fscore))
        # support
        print('support: {}'.format(support))

        return model
#----------------NB-----------------

    def nbtrain(organized_data):
        features = ['content', 'score', 'class']
        organized_data = organized_data.drop_duplicates()  # Remove duplicated rows
        organized_data.reset_index(drop=True, inplace=True)  # Remove dataframe indexes
        Y = organized_data['class'].values  # We fixe the column class in Y
        X = organized_data.drop(columns=['class'])  # We drop it in X

        X_train, X_test, Y_train, Y_test = train_test_split(organized_data['content'], Y, test_size=0.3)

        pipe = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('model', GaussianNB())])  # NB

        # Fitting the model
        model = pipe.fit(X_train.toarray(), Y_train.toarray())

        # Saving model
        pickle.dump(model, open('modelnb.sav', 'wb'))

        # Prediction of X_test
        print(Y_test)
        prediction = model.predict(X_test)
        print(prediction)

        #Evaluation
        precision, recall, fscore, support = score(Y_test, prediction)
        print("------------Evaluation-----------------")
        # Accuracy
        print("NB accuracy: {}%".format(round(accuracy_score(Y_test, prediction) * 100, 2)))
        #precision
        print('precision: {}'.format(precision))
        #recall
        print('recall: {}'.format(recall))
        #fscore
        print('fscore: {}'.format(fscore))
        #support
        print('support: {}'.format(support))

        return model


    # --- Prediction function --- #
    def predict(model, text):
        prediction = model.predict([text])
        return prediction

# --- Main function --- #
if __name__ == '__main__':
    # --- Getting and cleaning data --- #
    data = getDataFromDB()['content']
    string_text = {key: [putIntoString(value)] for (key, value) in data.items()}
    data_df = putDataInDataFrame(string_text)
    data_cleaning = lambda x: cleanData(x)

    # --- Organizing data in a pandas dataframe --- #
    cleaned_data = pd.DataFrame(data_df.content.apply(data_cleaning))
    organized_data = organizeData(data_df, getDataFromDB()['score'], getDataFromDB()['class'], cleaned_data)

    # --- Training data --- #
    #-----Building, and evaluating models + Test ------#
    print("-------------KNN--------------")
    modelknn = TrainingModels.knntrain(organized_data)
    prediction1 = TrainingModels.predict(modelknn, "Covid19 was created by china")
    prediction2 = TrainingModels.predict(modelknn, "many people in the UK have now had their first coronavirus vaccine")
    print("Covid19 was created by china: KNN said this is ")
    print(prediction1)
    print("many people in the UK have now had their first coronavirus vaccine: KNN said this is ")
    print(prediction2)
    print("-------------ANN--------------")
    modelann = TrainingModels.anntrain(organized_data)
    prediction1 = TrainingModels.predict(modelann, "Covid19 was created by china")
    prediction2 = TrainingModels.predict(modelann, "many people in the UK have now had their first coronavirus vaccine")
    print("Covid19 was created by china: ANN said this is ")
    print(prediction1)
    print("many people in the UK have now had their first coronavirus vaccine: ANN said this is ")
    print(prediction2)
    # print("-------------NB--------------")
    # modelknn = TrainingModels.nbtrain(organized_data)
    print("-------------DT--------------")
    modeldt = TrainingModels.dttrain(organized_data)
    prediction1 = TrainingModels.predict(modeldt, "Covid19 was created by china")
    prediction2 = TrainingModels.predict(modeldt, "many people in the UK have now had their first coronavirus vaccine")
    print("Covid19 was created by china: DT said this is ")
    print(prediction1)
    print("many people in the UK have now had their first coronavirus vaccine: DT said this is " )
    print(prediction2)
    print("-------------SVM--------------")
    modelsvm = TrainingModels.svmtrain(organized_data)
    prediction1 = TrainingModels.predict(modelsvm, "Covid19 was created by china")
    prediction2 = TrainingModels.predict(modelsvm, "many people in the UK have now had their first coronavirus vaccine")
    print("Covid19 was created by china: SVM said this is ")
    print(prediction1)
    print("many people in the UK have now had their first coronavirus vaccine: SVM said this is ")
    print(prediction2)
