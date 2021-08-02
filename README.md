# ANS Translation Service
Flask API for the ANS translation service and web interface. 

## Usage

To run with Python>=3.6:

```
pip install -r requirements.txt
python app.py
```

To run with docker:

```
docker build -t machine-translation-service .
docker run -p 5000:5000 -v /path/to/models:/app/data -it machine-translation-service
```

The front end should then become available at http://localhost:5000.

Call the service with curl:
```
curl --location --request POST 'http://localhost:5000/translate' \
--header 'Content-Type: application/json' \
--data-raw '{
 "text":"hello",
 "source":"en",
 "target":"fr"
}'
```
