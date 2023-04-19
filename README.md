# ANS Translation Service

## Demo & API
Web interface and Flask API for the ANS translation service

## Usage

To run with Python>=3.6:

```
pip install -r requirements.txt
python app.py
```

The front end should then become available at http://localhost:5000.

Call the service with curl:
```
curl --location --request GET 'http://localhost:5000/translate_with_parameters?text=cat&metric=True&boolMult=False' \
--header 'Content-Type: application/json'
```
