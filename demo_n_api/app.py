import os
import flask
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask import redirect, url_for, session
from translate import Translator
from config import *

from flask import Flask,render_template, request
from flaskext.mysql import MySQL

from flask_paginate import Pagination, get_page_parameter
from flask_bcrypt import Bcrypt

import re

app = Flask(__name__)
bcrypt = Bcrypt(app)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'mykey'

app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'ans'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

app.config["DEBUG"] = True # turn off in prod

mysql = MySQL()
mysql.init_app(app)

translator = Translator(MODEL_PATH)

@app.route('/')
@app.route('/index', methods=['GET'])
def home():
    # return render_template('index.html')
    if 'loggedin' in session:
        return render_template('home_advanced.html', username=session['username'])

        # User is loggedin show them the home page
    return render_template('home.html')

    # User is not loggedin redirect to login page
    #return redirect(url_for('login'))

@app.route('/docs/', defaults={'filename':'index.html'})
@app.route('/docs/<path:filename>')
def serve_sphinx_docs(filename) -> flask.send_from_directory:
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    return send_from_directory(directory=docs_dir, path=filename)

@app.route('/change_language', methods=["GET","POST"])
def change_language():
    trModel = request.form['trModel']
    translator.load_model(trModel)
    return "Machine translation service is up and running."

@app.route('/translate_with_parameters', methods=['GET','POST'])
def my_post():
    text = request.args.get('text')
    trModel = request.args.get('trModel')
    mult = "False"
    applyRules = "False"
    metric = "False"
    score = "0000"
    # print(mult)
    # print(request.form['boolMult'])
    if request.args.get('boolMult')=="True":
        mult = "True"
    if request.args.get('applyRules')=="True":
        applyRules = "True"
    if request.args.get('metric')=="True":
        metric = "True"
    if trModel=="fr":
        translation = translator.translate_reverse(text, mult, applyRules)
        result = {
            "output": translation
        }
    else:
        if metric=="True":
            translation, score, score2 = translator.translate_fairseq(text, mult, applyRules, metric)
            result = {
                "output": translation,
                "score": score,
                "score2": score2
            }
        else:
            translation = translator.translate_fairseq(text, mult, applyRules, metric)
            result = {
                "output": translation
            }

    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/translate', methods=['GET','POST'])
def my_form_post():
    text = request.form['text1']
    trModel = request.form['trModel']
    mult = "False"
    applyRules = "False"
    metric = "False"
    score = "0000"
    # print(mult)
    # print(request.form['boolMult'])
    if request.form['boolMult']=="True":
        mult = "True"
    if request.form['applyRules']=="True":
        applyRules = "True"
    if request.form['metric']=="True":
        metric = "True"
    if trModel=="fr":
        translation = translator.translate_reverse(text, mult, applyRules)
        result = {
            "output": translation
        }
    else:
        if metric=="True":
            translation, score, score2 = translator.translate_fairseq(text, mult, applyRules, metric)
            result = {
                "output": translation,
                "score": score,
                "score2": score2
            }
        else:
            translation = translator.translate_fairseq(text, mult, applyRules, metric)
            result = {
                "output": translation
            }

    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/store', methods = ['POST', 'GET'])
def store():
    if request.method == 'GET':
        return "Login via the login Form"

    if request.method == 'POST':
        eng = request.form['text1']
        trans = request.form['trans']
        user = session['username']
        sugg = request.form['sugg']
        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.execute(''' INSERT INTO suggestions VALUES(%s,%s,%s,%s)''',(eng,trans,sugg,user))
        conn.commit()
        # cursor.close()
        return f"Done!!"

@app.route('/suggestions')
def show_suggestions():
    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 20
    offset = page*limit - limit

    conn = mysql.connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * from suggestions")
    # result = cursor.fetchall()
    # total = len(result)
    # cursor.execute("SELECT * from suggestions ORDER by id DESC LIMIT %s OFFSET %S", (limit, offset))
    # data = cursor.fetchall()

    rows = []
    for row in cursor:
        rows.append(row)
    return render_template('suggestions.html', suggestions=rows)
    # pagination = Pagination(page=page, per_page=limit, total=total, record_name='suggestions')
    # return render_template('suggestions.html', pagination=pagination, suggestions=data)


@app.route('/login', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']

        # Check if account exists using MySQL
        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        # Fetch one record and return result
        # conn.commit()
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            # print(sha256_crypt.verify(account[2], hashed_pw))
            if bcrypt.check_password_hash(account[2], password):
                session['loggedin'] = True
                session['id'] = account[0]
                session['username'] = account[1]
                # Redirect to home page
                # return 'Logged in successfully!'
                return render_template('home_advanced.html', msg=msg)
            else:
                msg = 'Incorrect username/password!'
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('index.html', msg=msg)

@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   # return redirect(url_for('login'))
   return render_template('home.html')

# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Check if account exists using MySQL
        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            pw_hash = bcrypt.generate_password_hash(password)
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, pw_hash, email,))
            conn.commit()
            msg = 'You have successfully registered!'

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'

    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for loggedin users
@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/lang_routes', methods = ["GET"])
def get_lang_route():
    lang = request.args['lang']
    all_langs = translator.get_supported_langs()
    lang_routes = [l for l in all_langs if l[0] == lang]
    return jsonify({"output":lang_routes})

@app.route('/supported_languages', methods=["GET"])
def get_supported_languages():
    langs = translator.get_supported_langs()
    return jsonify({"output":langs})

# @app.route('/translate', methods=["POST"])
# def get_prediction():
#     # source = request.json['source']
#     # target = request.json['target']
#     text = request.json['text']
#     # translation = translator.translate(source, target, text)
#     translation = translator.translate_fairseq(text)
#     return jsonify({"output":translation})

if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host="0.0.0.0")
