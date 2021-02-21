from mongoengine import connect
from models import  Operation_Type, Data

# --- Connection to local database (MongoDB) ---
connect('ml_nlp_db', host='mongodb://localhost', alias='default')

# --- Test Function ---
def init_db():

    # --- Inserting some operation-types ---
    tokenization = Operation_Type(name = 'Tokenization')
    tokenization.save()

    stopWords = Operation_Type(name = 'Stop words')
    stopWords.save()

    lemmatization = Operation_Type(name = 'Lemmatization')
    lemmatization.save()

    stemming = Operation_Type(name = 'Stemming')
    stemming.save()

    posTagging = Operation_Type(name = 'Pos Tagging')
    posTagging.save()

    bagOfWords = Operation_Type(name = 'Bag of words')
    bagOfWords.save()

    TfIdf = Operation_Type(name = 'TF-IDF')
    TfIdf.save()

    word2Vec = Operation_Type(name = 'Word2Vec')
    word2Vec.save()

# --- Main function ---
if(__name__=='__main__'):
    init_db()