import pymongo

conn = "mongodb://system:system@ds127936.mlab.com:27936/heroku_njlc7ffz"

#'mongodb://localhost:27017'
client = pymongo.MongoClient(conn)

# Define database and collection
db = client.heroku_njlc7ffz

articles = db.tedtalk.find()

for article in articles:
	print(article)

