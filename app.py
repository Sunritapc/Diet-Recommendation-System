from math import*
from decimal import Decimal
from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import difflib
from flask_mysqldb import MySQL

app = Flask(__name__)  # application initiate karta hai
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'subofinal'
mysql = MySQL(app)

# model load kar raha hai pickle use karke
rfc = pickle.load(open('rfc.pkl', 'rb'))

dataset1 = pd.read_csv('Nutrients.csv')

dataset = pd.read_csv('input.csv')
dataset = dataset.drop(['VegNovVeg'], axis=1)
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)
fooditemlist = dataset['Food_items']
nutrientData = dataset.iloc[:, 4:].values
cosine_sim = linear_kernel(nutrientData, nutrientData)
smd = dataset.iloc[:, [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
titles = smd['Food_items']
indices = pd.Series(smd.index, index=smd['Food_items'])


def getfooditems(initialItem, smd):
    predictItems = []
    epoch = 0
    remainbf = initialItem

    while(True):
        if(epoch == 3):
            break
        if(checkRequirement(remainbf, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3, initialItem)):
            break
        bfpred = rfc.predict([remainbf])
        predictItems.append(bfpred)
        yo = smd['Food_items'].values == bfpred
        n = smd[yo]
        n = n.values.tolist()
        n = n[0][1:]
        remainbf = Diff(remainbf, n)
        epoch = epoch+1
    return(predictItems)


def nth_root(value, n_root):
    root_value = 1/float(n_root)
    return round(Decimal(value) ** Decimal(root_value), 3)


def checkRequirement(x, y, p_value, init):
    remain = nth_root(sum(pow(abs(a-b), p_value)
                      for a, b in zip(x, y)), p_value)
    initial = nth_root(sum(pow(abs(a-b), p_value)
                       for a, b in zip(init, y)), p_value)
    if(remain <= float(initial)*0.2):
        return True
    else:
        return False


def Diff(list1, list2):
    sublist = []
    for i in range(len(list1)):
        sublist.append(list1[i]-list2[i])
    return sublist


def get_recommendations(title, indices, cosine_sim, titles):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    food_items = [i[0] for i in sim_scores]
    return titles.iloc[food_items]


@app.route('/')  # ek pg se dusre pg jane ke lia-pg route
def home():
    return render_template('home.html')


@app.route('/blog')  # ek pg se dusre pg jane ke lia-pg route
def blog():
    return render_template('blog.html')


@app.route('/about')  # ek pg se dusre pg jane ke lia-pg route
def about():
    return render_template('about.html')


@app.route('/conatct')
def contact():
    return render_template('Contact.html')


@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', texxt=0)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == "POST":
        details = request.form
        username = details['username']
        email = details['email']
        password = details['password']
        cpassword = details['cpassword']
        if password != cpassword:
            return render_template('register.html', temp=1)
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO register(username, email,password) VALUES (%s, %s,%s)",
                    (username, email, password))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('index'))
    return render_template('register.html', temp=0)


@app.route('/mainform', methods=['POST'])  # data input dal rahe hai
def mainform():
    email = request.form['email']
    password = request.form['password']
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT email,password FROM REGISTER WHERE email = % s AND password = % s", (email, password, ))
    rv = cur.fetchall()
    mysql.connection.commit()
    cur.close()
    if(len(rv)):
        return render_template('main.html')
    else:
        return render_template('index.html', texxt=1)


@app.route('/nutrients', methods=['POST'])
def nutrients():
    # yaha hum input front end se lekar dal rahe hai back end me
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    height = int(request.form['height'])
    weight = int(request.form['weight'])
    goal = int(request.form['goal'])
    if gender == 0:
        bmr = 10 * (weight + goal) + 6.25 * height - 5 * age+5
    else:
        bmr = 10 * (weight + goal) + 6.25 * height - 5 * age - 161
    cal = bmr*float(request.form['active'])
    bmi = weight/((height*height)/10000)
    if bmi < 18.5:
        person = 'Underweight'
    elif bmi >= 18.5 and bmi <= 25:
        person = 'Healthy'
    elif bmi > 25:
        person = 'Overweight'

    fat = (cal*0.3)/9
    carbs = (cal*0.45)/4
    arr = dataset1.loc[(dataset1['Age'] == age)
                       & (dataset1['Gender'] == gender)].values
    required = [cal, fat, arr[0][2], arr[0][3], arr[0][4],
                arr[0][5], arr[0][6],	carbs, arr[0][7],	arr[0][8],	arr[0][9]]
    breakfastCalRequirment = [x * 0.37 for x in required]
    lunchCalRequirment = [x * 0.32 for x in required]
    dinnerCalRequirment = [x * 0.31 for x in required]

    breakfastrecom = getfooditems(breakfastCalRequirment, smd)
    lunchrecom = getfooditems(lunchCalRequirment, smd)
    dinnerrecom = getfooditems(dinnerCalRequirment, smd)
    breakfastalt = []
    lunchalt = []
    dinneralt = []
    for i in breakfastrecom:
        breakfastalt.append((get_recommendations(
            i[0], indices, cosine_sim, titles)[0:2]).tolist())
    for i in lunchrecom:

        lunchalt.append((get_recommendations(
            i[0], indices, cosine_sim, titles)[0:2]).tolist())

    for i in dinnerrecom:
        dinneralt.append((get_recommendations(
            i[0], indices, cosine_sim, titles)[0:2]).tolist())

    return render_template('nutrients.html', breakfastrecom=breakfastrecom, lunchrecom=lunchrecom, dinnerrecom=dinnerrecom, dinneralt=dinneralt, lunchalt=lunchalt, breakfastalt=breakfastalt, lenbreakfastrecom=len(breakfastrecom), lendinnerrecom=len(dinnerrecom), lenlunchrecom=len(lunchrecom), bmi=round(bmi, 1), cal=cal, person=person)


if __name__ == "__main__":
    app.run(debug=true)
