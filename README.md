# Topical-Sentiment-Analysis
ML model that recognises how much the text is related to data of a particular topic which the model is trained with. Modular structure of the code makes it easier to understand and modify it.

### How to use
#### Step 1 : Install requirements in either Pyhton2 or Python3
```pip install -r requirements.txt```
#### Step 2 : Download nltk_data
#### Step 3 : Execute TEST.py
```python TEST.py```

You can change the ```datasetPath``` according to your dataset filename and location from ```path.py``` .

You can also change the data processing flow of ```prepareDocuments(datasetPath,documentsPath,datasetTweetsPath)``` in ```classify.py```.

Note: If you are switching from python2 to python3 or vice versa, You need to delete all the ```.pickle``` files from pickled_classifiers and pickled_data. Then, execute TEST.py again.
