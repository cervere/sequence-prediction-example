#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import json


# In[53]:


class SubSet:
    def __init__(self, df):
        self.rawdata = df
        self.data = np.stack(df["data"]) if not df["data"].empty else np.array([])
        self.labels = np.stack(df["label"]) if not df["label"].empty else np.array([])

class DataSet:
    def __init__(self, vocabulary, sentence_length, validation_split=0.2, chunk_length=1):
        self.train = None
        self.validation = None
        self.test = None
        self.vocabulary = vocabulary
        self.output_labels = len(vocabulary)
        self.sentence_length = sentence_length
        self.v_split = validation_split
        self.chunk_length = chunk_length

    def load_data(self, traincsv, testcsv, blow_training_data=0, patterns=False):
        alltraindf = self.read_from_csv(traincsv, patterns=patterns)
        if blow_training_data > 0:
            alltraindf = pd.concat([alltraindf] * (blow_training_data + 1), ignore_index=True)
            alltraindf = alltraindf.sample(frac=1).reset_index(drop=True)
        alltestdf = self.read_from_csv(testcsv)
        t_split = 1 - self.v_split
        if t_split < 1 :
            traindf, validationdf = np.split(alltraindf.sample(frac=1), [int(t_split * len(alltraindf))])
        else : 
            traindf, validationdf = alltraindf.sample(frac=1), pd.DataFrame({'data' : [], 'label' : []})
        self.train = SubSet(traindf)
        self.validation = SubSet(validationdf)
        self.test = SubSet(alltestdf)
        return self

    def read_from_csv(self, csv, patterns=False):
        alldata = pd.read_csv(csv, names=["sentence"])
        if patterns :
            alldata_half = alldata.copy()
            alldata_half["sentence"] = pd.DataFrame(alldata_half["sentence"].apply(lambda x: x[:5]))
            alldata = alldata.append(alldata_half, ignore_index=True)
        alldata[['data', 'label']] = pd.DataFrame(alldata['sentence'].apply(self.split_sentence_ascii).tolist())
        # alldata["data"], alldata["label"] = zip(*alldata["sentence"].map(self.split_sentence_ascii))
        alldata['data'] = alldata['data'].apply(np.array)
        alldata["label"] = alldata["label"].apply(np.array)
        # databycol = pd.DataFrame(alldata["data"].to_numpy(), columns=[str(i) for i in range(8)])
        # print(databycol)
        return alldata

    def split_sentence_ascii(self, sentence):
        data = [self.get_binary(self.vocabulary[i], self.output_labels) for i in sentence[:-1]] + [np.zeros(self.output_labels) for i in
                                                                           range(self.sentence_length - len(sentence))]
        #         if SENTENCE_LENGTH - len(sentence) > 0:
        #             traindata
        data = np.array(data)
        data = np.reshape(data, (data.shape[0] // self.chunk_length, self.chunk_length * data.shape[1]))
        label = self.get_binary(self.vocabulary[sentence[-1]], self.output_labels)
        return data, label

    @staticmethod
    def get_binary(index, length):
        binary = np.zeros(length)
        if index > -1 and index < length: binary[index] = 1
        return binary

        

