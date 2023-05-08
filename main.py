from flask import *
import os
app = Flask(__name__)

import pickle
import pandas as pd
import numpy as np

auth = ''

def Normalize(X, Y):
    m = len(X) 
    n = len(Y) 
    L = [[0 for k in range(n+1)] for l in range(m+1)]
    r = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                L[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                L[i][j] = L[i-1][j-1] + 1
                r = max(r, L[i][j])
            else:
                L[i][j] = 0
    return (r/len(X))

@app.route("/")
def main():
    return render_template('index.html')
@app.route("/instagram", methods=["POST", "GET"])
def instagram():
    fcnt = 'a'
    if request.method == "POST":
        fcnt = request.form.get('content')
    n_div_len_uname = 1.0
    private = 0
    posts = 500
    followers = len(fcnt) * 50
    follows = 50
    z = pd.DataFrame(np.array([[n_div_len_uname,private,posts,followers,follows]]))
    with open('instagram/pickleOutput', 'rb') as f:
        mp = pickle.load(f)
    pickleTest = mp.predict(z)
    if request.method == "POST":
        if  pickleTest==0:
            auth='Real'
        else:
            auth='Fake'
        return render_template('instagram.html', auth=auth)
    return render_template('instagram.html')
@app.route("/facebook", methods=["POST", "GET"])
def facebook():
    fcnt = 'a'
    if request.method == "POST":
        fcnt = request.form.get('content')
    friends = len(fcnt) * 10
    following = 350
    community = 30
    sharedposts = 300
    cmt_per_post = 1.0
    likes_per_post = 5.0
    tags_per_post = 1.0
    z = pd.DataFrame(np.array([[friends, following, community, sharedposts, cmt_per_post, likes_per_post, tags_per_post]]))
    with open('facebook/pickleOutput', 'rb') as f:
        mp = pickle.load(f)
    pickleTest = mp.predict(z)
    if request.method == "POST":
        if  pickleTest==0:
            auth='Real'
        else:
            auth='Fake'  
        return render_template('facebook.html', auth=auth)
    return render_template('facebook.html')
@app.route("/twitter", methods=["POST", "GET"])
def twitter():
    fcnt = 'abcde'
    if request.method == "POST":
        fcnt = request.form.get('content')
    screen_name = "TwitterEng"
    name = "Twitter Engineering"
    name_wt = Normalize(name, screen_name)
    statuses_count = 20
    followers_count = 21
    friends_count = 250
    favourites_count = len(fcnt) - 5
    listed_count = 0
    z = pd.DataFrame(np.array([[name_wt,statuses_count,followers_count,friends_count,favourites_count,listed_count]]))
    with open('twitter/pickleOutputNew', 'rb') as f:
        mp = pickle.load(f)
    pickleTest = mp.predict(z)
    if request.method == "POST":
        if  pickleTest==0:
            auth='Real'
        else:
            auth='Fake'
        return render_template('twitter.html', auth=auth)
    return render_template('twitter.html')
@app.route("/fakenews")
def fakenews():
    return render_template('fakenews.html')
@app.route("/about")
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)