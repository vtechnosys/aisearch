import pandas as pd
import numpy as np
import os
import re
import operator
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import flask
from flask import session
from flask_session import Session
import pickle
import json
import pymysql
import time

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')


db = pymysql.connect(host='localhost',user='root',password='',db='searchengine')


app = flask.Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

uid = 1
#-------- MODEL GOES HERE -----------#

# print(df_news)
df_news = pd.read_csv('df_news_index.csv')
# query = "SELECT * FROM tbl_task"
# df_news = pd.read_sql(query, db)
with open("vocabulary.txt", "r") as file:
    vocabulary = eval(file.readline())

Tfidmodel =pickle.load(
    open('tfid.pkl', 'rb'))
traineddata = Tfidmodel.A #np.float16(Tfidmodel.A)

logstate = False;

def wordLemmatizer(data):
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    file_clean_k = pd.DataFrame()
    for index, entry in enumerate(data):

        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if len(word) > 1 and word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
                file_clean_k.loc[index, 'Keyword_final'] = str(Final_words)
                file_clean_k.loc[index, 'Keyword_final'] = str(Final_words)
                
    return file_clean_k

## Create vector for Query/search keywords


def gen_vector_T(tokens, tfidf):

    Q = np.zeros((len(vocabulary))) 
    x = tfidf.transform(tokens)
    for token in tokens[0].split(','):
        try:
            ind = vocabulary.index(token)
            Q[ind] = x[0, tfidf.vocabulary_[token]]
        except:
            pass
    return Q


def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

#-------- ROUTES GO HERE -----------#
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/login')
def login():
    return flask.render_template('login.html')

@app.route('/signin',methods=["POST"])
def signin():
    username = str(flask.request.form.get('username'))
    password = str(flask.request.form.get("password"))
    cursor = db.cursor()
    cursor.execute("select * from tbl_user where email ='"+username+"' and password='"+password+"'")
        
    if(cursor.rowcount>0):
        records = cursor.fetchall()
        uid = records[0]
        session['uid'] = uid[0]
        session['uname'] = uid[1]
        session['utype'] = uid[4]
        if(uid[4] == 'admin'):
            return flask.redirect('/dashboard')
        else:
            return flask.redirect('/user-dashboard')
    else:
        return "login Failed"

@app.route('/user-dashboard')
def user_dashboard():
    uid = session['uid']
    # print(uid)
    cursor = db.cursor()
    cursor.execute("select * from tbl_tickets where assign_user="+str(uid))
    trecords = cursor.rowcount
    cursor.execute("select * from tbl_tickets where t_status='pending' and assign_user="+str(uid))
    precords = cursor.rowcount
    cursor.execute("select * from tbl_tickets  where t_status='completed' and assign_user="+str(uid))
    crecords = cursor.rowcount
    return flask.render_template('user_dashboard.html',tcount = trecords,pcount = precords, ccount = crecords,uname=session['uname'],utype=session['utype'])

@app.route('/dashboard')
def dashboard():
    cursor = db.cursor()
    cursor.execute("select * from tbl_tickets")
    trecords = cursor.rowcount
    cursor.execute("select * from tbl_tickets where t_status='pending'")
    precords = cursor.rowcount
    cursor.execute("select * from tbl_tickets  where t_status='completed'")
    crecords = cursor.rowcount
    return flask.render_template('dashboard.html',tcount = trecords,pcount = precords, ccount = crecords,uname=session['uname'],utype=session['utype'])

@app.route('/adduser')
def register():
    return flask.render_template('register.html',uname=session['uname'])
@app.route('/add-task')
def task():
    cursor = db.cursor()
    cursor.execute("select * from tbl_user where utype<>'admin'")
    records = cursor.fetchall()
    return flask.render_template('addtask.html',users = records,uname=session['uname'],utype=session['utype'])


@app.route('/storetask',methods=['POST'])
def storetask():
    qst = str(flask.request.form.get('qst'))
    ans = str(flask.request.form.get('usr'))
    ptype = str(flask.request.form.get('ptype'))
    cursor = db.cursor()
    cursor.execute("insert into tbl_tickets(task,assign_user,t_priority,t_status,created_by) values('"+qst+"','"+ans+"','"+ptype+"','pending','1')")
    db.commit()
    return flask.redirect('/dashboard')

@app.route('/my-tasks')
def showmytask():
    uid = session['uid']
    cursor = db.cursor()
    cursor.execute("select * from tbl_tickets where assign_user = "+str(uid))
    records = cursor.fetchall()
    return flask.render_template('mytask.html',tasks = records,uname=session['uname'],utype=session['utype'])

@app.route('/show-task',methods=['GET'])
def showlog():
    tid = flask.request.args.get('tid')
    cursor = db.cursor()
    cursor.execute("select * from tbl_tickets where id = "+str(tid))
    records = cursor.fetchall()
    return flask.render_template('showticket.html',tasks = records,uname=session['uname'],utype=session['utype'])

@app.route('/show-log',methods=['GET'])
def showlogs():
    tid = flask.request.args.get('tid')
    cursor = db.cursor()
    cursor.execute("select * from tbl_task_log where task_id = "+str(tid))
    records = cursor.fetchall()
    cursor.execute("select * from tbl_tickets where id = "+str(tid))
    tasks = cursor.fetchall()
    return flask.render_template('showlog.html',tasks = tasks,records=records,uname=session['uname'],utype=session['utype'])

@app.route('/store-log',methods=['POST'])
def storelog():
    uid = session['uid']
    tid = flask.request.form.get('tid')
    res = flask.request.form.get('res')
    status = flask.request.form.get('status')
    cursor = db.cursor()
    cursor.execute("insert into tbl_task_log(task_id,response,assign_user_id) values("+str(tid)+",'"+str(res)+"',"+str(uid)+")")
    
    cursor.execute('update tbl_tickets set t_status="'+status+'" where id='+str(tid))
    db.commit()
    return flask.redirect('/user-dashboard')

@app.route('/view-tasks')
def showtask():
    cursor = db.cursor()
    cursor.execute("select * from tbl_tickets as t join tbl_user as u on t.assign_user = u.id")
    records = cursor.fetchall()
    return flask.render_template('showtask.html',tasks = records,uname=session['uname'],utype=session['utype'])
@app.route('/register', methods=['POST'])
def adduser():
    uname = str(flask.request.form.get('username'))
    email = str(flask.request.form.get('email'))
    utype = str(flask.request.form.get('utype'))
    password = str(flask.request.form.get('password'))

    cursor = db.cursor()
    cursor.execute("INSERT INTO `tbl_user`(`uname`, `email`, `password`, `utype`, `status`) VALUES('"+uname+"','"+email+"','"+password+"','"+utype+"','active')")
    db.commit();
    # return "INSERT INTO `tbl_user`(`uname`, `email`, `password`, `utype`, `status`) VALUES('"+uname+"','"+email+"','"+password+"','"+utype+"','active')"
    return flask.redirect('/view-users')

@app.route('/view-users')
def users():
    cursor = db.cursor()
    cursor.execute("select * from tbl_user")
    records = cursor.fetchall()
    return flask.render_template('showusers.html',users = records,uname=session['uname'],utype=session['utype'])
@app.route('/search', methods=["GET"])
def DrugFind():
    start = time.time()
    query = flask.request.args.get('query')

    cursor = db.cursor()
    cursor.execute("SELECT * FROM tbl_tickets as t JOIN tbl_task_log as l on t.id = l.task_id JOIN tbl_user as u ON t.assign_user = u.id WHERE MATCH (task) AGAINST ('"+query+"' IN NATURAL LANGUAGE MODE)")
    records = cursor.fetchall()
    preprocessed_query = preprocessed_query = re.sub(
        "\W+", " ", query.lower()).strip()
    tokens = word_tokenize(str(preprocessed_query))
    q_df = pd.DataFrame(columns=['q_clean'])
    q_df.loc[0, 'q_clean'] = tokens
    
    q_df['q_clean'] = wordLemmatizer(q_df.q_clean)
    q_df = q_df.replace(to_replace="'", value='', regex=True)
    q_df = q_df.replace(to_replace="\[", value='', regex=True)
    q_df = q_df.replace(to_replace=" ", value='', regex=True)
    q_df = q_df.replace(to_replace='\]', value='', regex=True)

    d_cosines = []
    tfidf = TfidfVectorizer(vocabulary=vocabulary , dtype=np.float32)
    tfidf.fit(q_df['q_clean'])
    query_vector = gen_vector_T(q_df['q_clean'], tfidf)
    #query_vector = np.float16(query_vector)
    for d in traineddata:
        d_cosines.append(cosine_sim(query_vector, d))
    out = np.array(d_cosines).argsort()[-10:][::-1]
   
    d_cosines.sort()

    a = pd.DataFrame()
    for i, index in enumerate(out):
        a.loc[i, 'Subject'] = df_news['Subject'][index]
        a.loc[i, 'content'] = df_news['content'][index]
    for j, simScore in enumerate(d_cosines[-10:][::-1]):
        a.loc[j, 'Score'] = simScore
    a = a.sort_values(by='Score', ascending=False)
    js = a.to_json(orient='index')
    js =js.replace('[', '').replace(']', '')
    ls = js.split('},')

    l = [re.sub(r'\"[0-9]\":', '', l) for l in ls]
    l[0] = re.sub(r'^{{1}', '', l[0])      
    l = [re.sub(r'^,{1}', '', l) for l in l]
    l = [ls+'}' for ls in l]
    l[9] = l[9].replace('}}', '')
    lsDrug =[]
    for txt in l:
        tx =json.loads(txt)
        lsDrug.append(tx)
    # response = app.response_class(
    #     response=json.dumps(lsDrug),
    #     status=200,
    #     mimetype='application/json'
    # )
    end = time.time()
    t = end-start
    tm = "{:.2f}".format(round(t, 2))
    return flask.render_template('search.html',records = records,tm=tm,search = lsDrug,query=query,uname=session['uname'],utype=session['utype'])

    # return flask.jsonify(lsDrug) 


if __name__ == '__main__':
    'Connects to the server'
    HOST = '127.0.0.1'
    PORT = 5000      #make sure this is an integer
    # export FLASK_ENV=development
    app.run(debug=True)
