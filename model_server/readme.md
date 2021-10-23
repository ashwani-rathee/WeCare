This is the breast tumor segmentation rest api flask server that is supposed to be deployed on heroku or gcp.

There is only essential endpoint that's is `/predict` that is supposed to be used to predict the segmentation of the brain tumor. Predict works on two end points:

- `/predict`: This is the endpoint that is used to segment the breast tumor.


#### Local run:

```
cd model_server
conda create -n breast-tumor-segmentation python==3.8
pip install -r requirements.txt
FLASK_ENV=development FLASK_APP=app.py flask run
```

That would hopefully output something like below:
```
(base) ashwani@user:~/annotater/segment_server$ FLASK_ENV=development FLASK_APP=app.py flask run
 * Serving Flask app "app.py" (lazy loading)
 * Environment: development
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with inotify reloader
 * Debugger is active!
 * Debugger PIN: 671-247-288
```
This means that the server is running on localhost:5000 and is ready to take up request on point `/predict`

Server run can be tested by used request_test.py script.
```
python request_test.py
```

which will hopefully return 
```
(py36) ashwani@user:~/breast_cancer_awareness/model_server$ python request_test.py
[[False False False ... False False False]
 [False False False ... False False False]
 [False False False ... False False False]
 ...
 [False False False ... False False False]
 [False False False ... False False False]
 [False False False ... False False False]]
```


#### GCP deployement:

Create a project by name brain-tumor-segment-api and enable app engine+other api use by adding the billiing account(also use those credits by verifying debit card). Then start the cloud shell on top right corner of the project page. Then do the following steps in shell. app.yaml file is essential to the deployement and not Procfile.
```
https://github.com/ashwani-rathee/breast_cancer_awareness.git
cd breast_cancer_awareness/model_server
gcloud app deploy
```

Then it will prompt you to add some yes/no options and the whole log will look something like below:
```
ab669522@cloudshell:~/breast_cancer_awareness/model_server $ gcloud app deploy

```
Now the website can be found there on link provided above.