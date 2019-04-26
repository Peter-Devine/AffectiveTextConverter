import argparse
import os
import pandas as pd
import numpy as np
import re

# Takes input and output directories as arguments
parser=argparse.ArgumentParser()
parser.add_argument('--input', default=".", help='The file path of the unzipped Facebook VA dataset')
parser.add_argument('--output', default="./data", help='The file path of the output dataset')
parser.add_argument('--vad_bin_num', default="7", help='Number of bins to separate the valence values into (if you edit this you need to also edit the BERT classifier code)')

args = parser.parse_args()
INPUT_PATH = args.input
OUTPUT_PATH = args.output
VAD_BIN_NUM = int(args.vad_bin_num)

# Make the output directory if it does not currently exist
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

def strip_tags(string):
    return(str(re.search('>(.*)<', string).group(1)))

def take_id(string):
    return(int(string.split('"')[1]))

bins = []

for dataframe_type in ["trial", "test"]:

    f = open(INPUT_PATH + "/affectivetext_" + dataframe_type + ".xml", "r")

    fl = f.readlines()

    f.close()

    ids = pd.Series(fl[1:(len(fl)-1)]).apply(take_id)
    texts = pd.Series(fl[1:(len(fl)-1)]).apply(strip_tags)

    combined_df = pd.DataFrame({"id": ids, "text": texts})

    emotion = pd.read_csv(INPUT_PATH + "/affectivetext_" + dataframe_type + ".emotions.GOLD", sep=" ", names=["id"]+emotions)
    valence = pd.read_csv(INPUT_PATH + "/affectivetext_" + dataframe_type + ".valence.GOLD", sep=" ", names=["id", "valence"])

    affective_text = combined_df.join(emotion, lsuffix='text').join(valence, rsuffix='valence')

    affective_text["emotion"] = affective_text[emotions].idxmax(axis=1)
    
    bin_labels = ["V"+str(bin_label_index+1) for bin_label_index in range(VAD_BIN_NUM)]
    
    if dataframe_type == "trial":
        
        fraction = 0.2

        np.random.seed(seed=42)

        dev_indices = np.random.choice(affective_text.index, size=int(round(fraction*affective_text.shape[0])), replace=False)
        train_indices = affective_text.index.difference(dev_indices)

        affective_text_train = affective_text.loc[train_indices,:]
        affective_text_dev = affective_text.loc[dev_indices,:]
        
        binned_data = pd.cut(affective_text_train["valence"], bins=VAD_BIN_NUM, retbins=True)
        bins = binned_data[1]

        affective_text_train["V_binned"] = pd.cut(affective_text_train["valence"], bins=bins, labels=bin_labels)
        affective_text_dev["V_binned"] = pd.cut(affective_text_dev["valence"], bins=bins, labels=bin_labels)
        affective_text_train.reset_index(drop=True).to_csv(OUTPUT_PATH+"/train.tsv", sep='\t', encoding="utf-8")
        affective_text_dev.reset_index(drop=True).to_csv(OUTPUT_PATH+"/dev.tsv", sep='\t', encoding="utf-8")
    else:
        affective_text["V_binned"] = pd.cut(affective_text["valence"], bins=bins, labels=bin_labels)
        affective_text.to_csv(OUTPUT_PATH+"/test.tsv", sep='\t', encoding="utf-8")