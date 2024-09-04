# ANS Translation Service

This is the Github repo containing all work regarding the machine translation pipeline: scripts, docs and resources.

## Demo & API

You can run the web interface and Flask API for the ANS translation service via Docker or Python. The Docker option also offers the necessary database to store user accounts and propose suggestions when available.

### Docker

To run:

```
sudo apt-get install docker.io
cd demo_n_api
sudo docker compose up --build
```

The front end should then become available at http://localhost. If you don't run the demo locally use the online tool http://anstranslation2.ddns.net.

### Python

Create a new virtualenv with Python 3.9, and then run:

```
pip install -r requirements.txt
cd demo_n_api/app
python app.py
```

## Scripts
To run postprocessing rules check the `conf_processing.py` file.
