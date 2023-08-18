from datetime import datetime


def insert_one_(collection, data):
    data['creation_date'] = datetime.now()

    # Insert the modified data document into the collection
    collection.insert_one(data)
