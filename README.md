# Installations
In order to run the codes in this project, the following libraries must be installed:
1. Pandas
2. Numpy
3. Sci-kit Learn
4. Flask
5. SQL Alchemy
6. Plotly
7. NLTK

# Motivation
This project was done to complete the requirements for Udacity's Data Scientist Nanodegree. Using text data from Figure-8, a company specializing in data analytics and machine learning, the purpose was to classify messages that were created during a disaster into 36 categories to help in aid efforts.

# Files
The project is divided into 3 folders: one for data and data processing; another one is for building a machine learning pipeline; and the third is for the web app.

### Files in the Data Folder
1. Messages data: disaster_messages.csv
2. Categories data: disaster_categories.csv
3. SQL Database: DisasterResponse.db
4. Jupyter notebook for building ETL pipeline: ETL Pipeline Preparation.ipynb
5. Python script for processing the data: process_data.py

### Files in the Models Folder
1. Jupyter notebook for building a machine learning pipeline: ML Pipeline Preparation.ipynb
2. Python script for training the classifier: train_classifier.py
3. A pickle file that contains the trained model: classifier.pkl

### Files in the App Folder
1. Python script for running the web app: run.py
2. templates folder that contains 2 HTML files for the app front-end: go.html and master.html


# Results
The final output of the project is an interactive web app that takes a message from the user as an input and then classifies it.

# Acknowledgement
Thanks to Udacity for providing guidance to complete the project and thanks to Figure-8 for providing the data
