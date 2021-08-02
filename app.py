import os
from flask import Flask, request, jsonify, render_template
from translate import Translator
from config import *

app = Flask(__name__)
translator = Translator(MODEL_PATH)

app.config["DEBUG"] = True # turn off in prod

@app.route('/')
@app.route('/index', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/change_language', methods=['GET','POST'])
def change_language():
    #trModel = request.args.get('trModel')
    trModel = request.form['trModel']
    result = translator.load_model(trModel)
    return jsonify({"output":result})

#i @app.route('/', methods=["GET"])
# def health_check():
#     """Confirms service is running"""
#     return "Machine translation service is up and running."

@app.route('/translate', methods=['GET','POST'])
def my_form_post():
    text1 = request.form['text1']
    #trModel = request.form['trModel']
    # text = request.args.get('text1')
    multiple = request.form['boolMult']
    # combine = do_something(text1,text2)
    # combine = get_prediction()
    # translation = translator.translate('en', 'fr', text1)
    translation = translator.translate_fairseq(text1, multiple)
    result = {
        "output": translation
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

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
    app.run(host="0.0.0.0", port='8000', debug=False)

    #host="0.0.0.0" will make the page accessable
                            #by going to http://[ip]:5000/ on any computer in
                            #the network.
