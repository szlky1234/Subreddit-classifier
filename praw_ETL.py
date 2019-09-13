import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize
import praw
import fasttext
from fasttext import util
import sys
import tensorflow as tf
import time

class Run:
    '''
    variables
    '''
    clientId = '0fg6j0-Lq23T6g'
    clientSecret = 'CtUZWmFwWKIRLSkpPKbajeBcD1U'
    userAgent = 'test app'
    reddit = praw.Reddit(client_id=clientId, client_secret=clientSecret, user_agent=userAgent)

    top_subreddits = {'Music': 1,
                      'AskReddit': 2,
                      'books': 3,
                      'worldnews': 4,
                      'Science': 5,
                      'food': 6,
                      'gaming': 7,
                      'IAmA': 8,
                      'technology': 9,
                      'movies': 10}

    tag_filters = ['IN', 'CC', 'DT']
    punc_filters = ["'", '"', ".", ",", "/", "-", "--", "(", ")", "('", ")'", "!')", ":", ";", "!'", "?'", ".)", ").",
                    "--'", '\'"', '--"', '"--', ",'", ".'", "*", ",)", '!"', "'--", '."', '[', ']', '’', '”', '“', ]

    file = "crawl-300d-2M-subword.bin"

    model = fasttext.load_model(file)
    output = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    lstm_model = None

    '''
    There should be something here
    '''
    def __init__(self):
        pass

    '''
    checks validity by ord()
    '''
    def is_valid_char(self, char):
        if len(char) == 1:
            return ((97 <= ord(char) <= 122)
                    or (48 <= ord(char) <= 57)
                    or (64 <= ord(char) <= 90)
                    or (ord(char) == 33)
                    or (ord(char) == 63)
                    or (ord(char) == 58))
        else:
            return True

    '''
    Converts an index to one-hot
    '''
    def to_onehot(self, index):
        x = np.zeros(10)
        np.put(x, index - 1, 1)
        return x

    '''
    gets top_subreddit key by value
    '''
    def get(self, val):
        return list(self.top_subreddits.keys())[list(self.top_subreddits.values()).index(val)]

    '''
    uses nltk.pos_tag to generate tagged chunks of the original sentence, filters for valid ascii characters and various 
    other attributes. 
    returns a sentence vector representation of the processed sentence
    '''

    def retrieve_and_vectorize(self, post_title):
        tokenized = nltk.pos_tag(word_tokenize(' '.join(post_title.split('/'))))
        filtered = list(filter(lambda x: (x[1] not in self.tag_filters and self.is_valid_char(x[0])), tokenized))
        for i, word in enumerate(filtered):
            filtered[i] = word[0]

        # print(' '.join(filtered))
        return self.model.get_sentence_vector(' '.join(filtered))

    '''
    retrieves top of all time posts (1,000) from each sub of top_subreddits, preprocesses each post title, bipartitions
    output, and populates x_train, y_train, x_test, y_test 
    '''
    def preprocess(self):
        for subreddit in self.top_subreddits.keys():
            counter = 0
            posts = self.reddit.subreddit(subreddit).top(time_filter="all", limit=1000)
            for post in posts:
                counter += 1
                if counter > 10:
                    self.output.append([post.title, post.subreddit.display_name,
                                        self.retrieve_and_vectorize(post.title),
                                        self.top_subreddits[subreddit]])
                    if counter % 20 == 0:
                        print("post number: " + str(counter) + " " + post.title)
            print(subreddit + " retrieved")
        np.random.shuffle(self.output)
        for index, data in enumerate(self.output):
            if index > np.floor(len(self.output) * 0.8):
                self.x_test.append([data[2]])
                self.y_test.append(self.to_onehot(data[3]))
            else:
                self.x_train.append([data[2]])
                self.y_train.append(self.to_onehot(data[3]))

        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)

        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
        print("all subs retrieved")
        print("x_train size: " + str(len(self.x_train)) + " x_test size: " + str(len(self.x_test)))
        print("y_train size: " + str(len(self.y_train)) + " y_test size: " + str(len(self.y_test)))

        # print(len(self.x_train[0]))
        # print(self.x_train[0].shape)
        # print(self.x_train[0])
        #
        # print(len(self.y_train[0]))
        # print(self.y_train[0].shape)
        # print(self.y_train[0])

        df = pd.DataFrame(self.output, columns=["post name", "subreddit", "embedding", "subnumber"])
        print(df.head())
        # df.to_csv("out.csv", encoding='utf-8', index=False)

    '''
    Begins model.fit based on x_train, y_train and saves model as praw_model
    '''
    def train(self):
        print("Beginning training ...")
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(128, activation='relu', input_shape=[1, 300], return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.LSTM(128, activation='relu', input_shape=[1, 300]))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(self.x_train, self.y_train, epochs=300)

        val_loss, val_acc = model.evaluate(self.x_test, self.y_test)
        print("Training complete ")
        print("val loss and val acc: ")
        print(val_loss, val_acc)
        model.save('praw_model')
        self.lstm_model = tf.keras.models.load_model('praw_model')
        print('Model saved')

    '''
    Runs predictions on x_test against y_test
    prints incorrect labels and accuracy
    '''
    def test(self):
        predictions = self.lstm_model.predict(self.x_test)
        accurate = 0

        for index, item in enumerate(predictions):
            # predictions
            if int(np.argmax(item)) + 1 == self.output[int(np.floor(len(self.output) * 0.8)) + index + 1][3]:
                accurate += 1

        for index, item in enumerate(predictions):
            if index > 15:
                break
            else:
                print("original post: ")
                print(self.output[int(np.floor(len(self.output) * 0.8)) + index + 1][0])
                print("original sub: ")
                print(self.output[int(np.floor(len(self.output) * 0.8)) + index + 1][1])
                print("predicted result: ")
                print(item)
                print("predicted sub: ")
                print(self.get(np.argmax(item) + 1))
                print("\n")

        print(str(accurate) + ' out of: ' + str(len(self.x_test)))
        print('accuracy of: ' + str(accurate / len(self.x_test)))

    '''
    This project is not engineered very well
    load_model loads the hard-coded praw_model into run instance
    '''
    def load_model(self):
        self.lstm_model = tf.keras.models.load_model('praw_model')

    '''
    vectorizes post, and runs prediction
    '''
    def predict(self, post):
        prediction = self.lstm_model.predict(np.array([[self.retrieve_and_vectorize(post)]]))
        ind = list(np.argpartition(prediction[0], -3)[-3:])
        for i in ind:
            print('Predict sub: ' + self.get(i + 1))
            print('Predict certainty: ' + str(prediction[0][i]))

run = Run()

while True:
    query = input("Query: \n")
    if query == 'exit':
        exit()
    elif query == 'load':
        run.load_model()
        print('model loaded!')
    elif query == 'predict':
        post_to_predict = input('Enter the post to be predicted\n')
        run.predict(post_to_predict)
    elif query == 'train':
        run.preprocess()
        run.train()
        run.test()

