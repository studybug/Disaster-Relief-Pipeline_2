# Disaster-Relief-Pipeline


## Table of contents
* [General info](#general-info)
* [General details](#General-details)
* [Technologies](#technologies)
* [Setup](#setup)

## General info

In this project, I create a machine learning pipeline to analyze real disaster data from Figure Eight with the intent that these message categories will be viewed and sent to an appropriate disaster relief agency. My project also included a web app where an emergency worker can input a new message and get classification results in several categories. 

## General details
The data came in this format: disaster_categories.csv, disaster_messages.csv
The project was cleaned, filtered, and categorized into two main files in a jupyternotebook: ETL Pipeline Preparation (1).ipynb, ML Pipeline Preparation (1).ipynb
The project was reorganized in a repetable execution for a terminal: process_data.py, train_classifier.py, run.py

Below are a few screenshots of the web app.

## Technologies
Project is created with:
* Python version: 3.7
* Jupyter Notebooks
* Terminal, Linux

	
## Setup
This project requires file:
* Data: disaster_categories.csv, disaster_messages.csv, DisasterResponse.db, process_data.py
* Model: classifier.pkl, train_classifier.py
* App: run.py, [templates]

* app
	* template
	* master.html  # main page of web app
	* go.html  # classification result page of web app
	* run.py  # Flask file that runs app
* data
	* disaster_categories.csv  # data to process 
	* disaster_messages.csv  # data to process
	* process_data.py
	* InsertDatabaseName.db   # database to save clean data to

* models
	* train_classifier.py
 	* classifier.pkl  # saved model 



To run the file on a terminal:

```python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db```

```python train_classifier.py ../data/DisasterResponse.db classifier.pkl```

Type in the command line:

```python run.py```

Then Go to http://0.0.0.0:3001/ or your use command env|grep WORK with https://SPACEID-3001.SPACEDOMAIN

