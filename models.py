from mongoengine import Document
from mongoengine.fields import (
    StringField, IntField
)

# --- Operation_Type model for operation_type collection --- #
class Operation_Type(Document):
    meta = {'collection' : 'operation_type'}
    name = StringField()

# --- Data model for data collection --- #
class Data(Document):
    meta = {'collection' : 'data'}
    Source = StringField()
    Title = StringField()
    News = StringField()
    language = StringField(default='english')
    Score = IntField()
    classe = IntField()