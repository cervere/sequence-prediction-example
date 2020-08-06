# Sequence Prediction Example
Dataset is not published here, but for an idea, training is done on sentences that look like **AEDF234TH** and testing is done on sentences like **ASE452DG?**.


### Requirements

Information about the Jupyter notebooks used : 

```
notebook server is: 5.7.0
Python 3.5.2 
IPython 7.0.1 
```
**Data :**
Place the directory `data` in the root directory, where the jupyter notebooks are.
The `data` folder is expected to have atleast 2 files : `train.csv` and `answers.csv`

```
root_dir
    |
    |
    --data
        |
        -- train.csv
        -- answers.csv      
```
The root directory contains the following Jupyter notebooks : 
 - [AlienChatDataSet](AlienChatDataSet.ipynb) : exploring the DataSet (formed from the files in `data` folder)
 - [AlienChat_LSTM_Keras](AlienChat_LSTM_Keras.ipynb) : LSTM implementation using Keras
 - [NN_AlienChat](NN_AlienChat.ipynb) : Basic Neural Network implementation, like an image classification problem
 
### Dependencies
```
tensorflow==2.3.0
numpy==1.18.5
pandas==0.24.2
```
