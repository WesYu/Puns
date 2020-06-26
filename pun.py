from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
def main(s):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(s)
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    filtered_words = pos_tag(filtered_words)
    top_score = 0
    length = len(filtered_words)
    for word in filtered_words:
        w_score = 0
        w = list(word)
        new_list = []
        for row in filtered_words:
            new_list.append(list(row))
        for i in range(length-1):
            p1 = filtered_words[:i+1]
            p2 = filtered_words[i+1:]
            w_score = max(w_score, score(w,p1,p2))
        if w_score>=top_score:
            top_score = w_score
            pun = w[0]
    new_word = ''
    for syn in wn.synsets(pun):
        for l in syn.lemmas():
            if l.name().lower() != pun.lower():
                new_word = l.name()
    return s.replace(pun, new_word, 1)
    
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''

def disamb(w, p):
    #for w, a word with pos tag, find the synset most similar to p
    #which is a set of words with pos tags
    word = w[0]
    tag = w[1]
    best_score = 0
    best_syn = wn.synset('dog.n.01')
    for synset in wn.synsets(word, pos=get_wordnet_pos(tag)):
        total_score = 0
        for row in p:
            total_score = total_score+syn_v_word(synset, row)
        avg_score = total_score/len(p)
        if avg_score>=best_score:
            best_score = avg_score
            best_syn = synset
    return best_score, best_syn
                
def syn_v_word(syn1, w):
    #average similarity between a synset and a word with pos tag
    word = w[0]
    tag = w[1]
    syn = wn.synsets(word, pos=get_wordnet_pos(tag))
    total_score = 0
    if not syn:
        return 0
    for syn2 in syn:
        similarity = wn.path_similarity(syn1,syn2)
        if similarity is None:
            similarity = 0
        total_score = total_score+similarity
    return total_score/len(syn)

def score(w, p1, p2):
    alpha = 1
    score1, syn1 = disamb(w, p1)
    score2, syn2 = disamb(w, p2)
    #incongruity
    incon = -wn.path_similarity(syn1, syn2)
    return alpha*(score1+score2)+incon

import pandas as pd
import numpy as np
import xml.etree.ElementTree as et

#match data with label
label_df = pd.read_csv('C:/Users/lenovo/desktop/label.gold',header=None,sep='\t')
label_df.columns = ['id','label']
data_tree = et.parse('C:/Users/lenovo/desktop/data.xml')
root = data_tree.getroot()

cols = ['sentence','label']
rows = []

for text in root.findall('text'):
    text_id = text.attrib.get('id')
    sentence = ''
    for word in text.findall('word'):
        sentence = sentence+word.text+' '
    label = label_df.loc[label_df['id']==text_id,'label'].values[0]
    rows.append({"sentence":sentence,'label':label})

df = pd.DataFrame(rows,columns=cols)

new_df = pd.DataFrame(columns=['before','after'])
for index, row in df.iterrows():
    if row['label']==1:
        pun = row['sentence']
        new_row = {'before': pun, 'after': main(pun)}
        new_df = new_df.append(new_row, ignore_index=True)