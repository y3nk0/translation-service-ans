import os

import flask
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask import redirect, url_for, session
from translate.translate import Translator
from translate.config import *

from flask_mysqldb import MySQL

from flask_paginate import get_page_parameter
from flask_bcrypt import Bcrypt
from flask_swagger_ui import get_swaggerui_blueprint

import re

app = Flask(__name__)
bcrypt = Bcrypt(app)

SWAGGER_URL = "/swagger"
API_URL = "/static/swagger.json"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Access API'
    }
)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'mykey'

app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'ans'
app.config['MYSQL_DATABASE_HOST'] = 'db'
app.config['MYSQL_ROOT_PASSWORD'] = 'root'

app.config['MYSQL_HOST'] = 'db'  # Use the service name defined in docker-compose
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'ans'

app.config["DEBUG"] = True # turn off in prod
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

mysql = MySQL(app)

translator = Translator(MODEL_PATH)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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


@app.route('/docs/', defaults={'filename': 'index.html'})
@app.route('/docs/<path:filename>')
def serve_sphinx_docs(filename) -> flask.send_from_directory:
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    return send_from_directory(directory=docs_dir, path=filename)


@app.route('/change_language', methods=["GET", "POST"])
def change_language():
    tr_model = request.form['trModel']
    translator.load_model(tr_model)
    return "Machine translation service is up and running."


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Here, you can save the file or process it
        file.save(UPLOAD_FOLDER + "/" + file.filename)
        return 'File successfully uploaded'


@app.route('/translate_file', methods=['POST'])
def translate_file():
    file_name = request.form['filename2'].split("\\")[-1]
    if file_name != "":
        f = open(UPLOAD_FOLDER+"/"+file_name, 'r')
        lines = f.readlines()
        f.close()
        mult, apply_rules, metric = False, False, False
        f = open('tmp/'+file_name.strip(".txt")+'_translated.txt', 'w')
        for line in lines:
            text = line.strip()
            translation = translator.translate_fairseq(text, mult, apply_rules, metric)
            f.write(translation+"\n")
        f.close()
        # Return the file for download
        return send_file('tmp/'+file_name.strip(".txt")+'_translated.txt', as_attachment=True)


@app.route('/upload_translate_file', methods=['POST'])
def upload_translate_file():
    file = request.files['file']
    lines = file.readlines()
    mult, apply_rules, metric = False, False, False
    f = open('tmp/' + file.filename.strip(".txt") + '_translated.txt', 'w')
    for line in lines:
        text = line.strip()
        translation = translator.translate_fairseq(text, mult, apply_rules, metric)
        f.write(translation + "\n")
    f.close()
    return send_file('tmp/'+file.filename.strip(".txt")+'_translated.txt', as_attachment=True)


@app.route('/translate', methods=['GET', 'POST'])
def my_post():
    metric = 'False'
    mult = 'False'
    metric = 'False'
    apply_rules = 'False'
    if request.method == 'POST':
        if request.headers['Content-Type'] == 'application/json':
            data = request.get_json()
            text = data['text']
            trModel = data['trModel']
            mult = data['boolMult']
            apply_rules = data['applyRules']
            metric = data['metric']
        else:
            text = request.form['text']
            trModel = request.form['trModel']
            mult = request.form['boolMult']
            apply_rules = request.form['applyRules']
            metric = request.form['metric']
    else:
        text = request.args.get('text')
        trModel = request.args.get('trModel')
        if request.args.get('boolMult') == "True":
            mult = "True"
        if request.args.get('applyRules') == "True":
            apply_rules = "True"
        if request.args.get('metric') == "True":
            metric = "True"

    suggestion = get_suggestion(text)

    if trModel == "fr":
        translation = translator.translate_reverse(text, mult, apply_rules)
        result = {
            "output": translation
        }
    else:
        if metric == "True":
            translation, score, score2 = translator.translate_fairseq(text, mult, apply_rules, metric)
            if suggestion:
                result = {
                    "output": translation,
                    "score": score,
                    "score2": score2,
                    "suggestion": suggestion
                }
            else:
                result = {
                    "output": translation,
                    "score": score,
                    "score2": score2
                }
        else:
            translation = translator.translate_fairseq(text, mult, apply_rules, metric)
            if suggestion:
                result = {
                    "output": translation, "suggestion": suggestion
                }
            else:
                result = {
                    "output": translation
                }

    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route('/store', methods=['POST'])
def store():
    if request.method == 'POST':
        eng = request.form['text']
        trans = request.form['trans']
        if 'user' in request.form:
            user = request.form['user']
        else:
            user = session['username']
        sugg = request.form['sugg']
        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.execute(''' INSERT INTO suggestions VALUES(%s,%s,%s,%s)''', (eng, trans, sugg, user))
        conn.commit()
        # cursor.close()
        return f"Done!!"


@app.route('/get_suggestion', methods=['POST'])
def get_suggestion(text):
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT * FROM suggestions WHERE eng = %s', (text,))
    result = cursor.fetchone()
    if result:
        return result[1]
    return ""


@app.route('/suggestions')
def show_suggestions():
    page = request.args.get(get_page_parameter(), type=int, default=1)
    limit = 20
    offset = page*limit - limit

    cursor = mysql.connection.cursor()
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
        cursor = mysql.connection.cursor()
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
        conn = mysql.connection()
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
            # Account doesn't exist and the form data is valid, now insert new account into accounts table
            pw_hash = bcrypt.generate_password_hash(password)
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, pw_hash, email,))
            conn.commit()
            msg = 'You have successfully registered!'

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'

    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)


@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor()
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
    return jsonify({"output": lang_routes})


@app.route('/supported_languages', methods=["GET"])
def get_supported_languages():
    langs = translator.get_supported_langs()
    return jsonify({"output": langs})


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host="0.0.0.0", port=5000)
