# ez_baes
Fit bayesian MCMC models in the cloud from your browser

# Dev
Setup
```
brew install redis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export FLASK_APP=app/views.py
```

start redis server
```
redis-server
```

start huey consumer
```
cd app
huey_consumer.py iris.huey
```

run flask app
```
flask run
```
