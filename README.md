# ANS Translation Service

Create a new virtualenv with Python 3.7.13, and then run:

```
pip install -r requirements.txt
```

## Scripts
To run postprocessing rules check the `conf_processing.py` file.

## Demo & API
Web interface and Flask API for the ANS translation service

### Usage

To run:

```
python app.py
```

The front end should then become available at http://localhost:5000. If you don't run the demo locally use the online tool http://anstranslation.ddns.net:5000/.

Call the service with curl:
```
curl --location --request GET 'http://localhost:5000/translate_with_parameters?text=cat&metric=True&boolMult=False' \
--header 'Content-Type: application/json'
```
