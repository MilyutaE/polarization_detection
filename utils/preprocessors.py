import pandas as pd
import numpy as np

import pymorphy2
from stop_words import get_stop_words

from tqdm import tqdm
import re, os, sys, codecs, json
from abc import abstractmethod
from typing import List


class Preprocessing():
    def __init__(self, *args, **kwargs):
        self.dataset = None
        self.processed_ds = None
    
    @abstractmethod
    def run(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def read_data(self,  file_path:str, test=False)-> 'Json_dataset':
        pass
    

class TonalPreprocessing(Preprocessing):
    def __init__(self, opinion_words):
        assert "word" in opinion_words.columns and "rating" in opinion_words.columns, 'opinion_words  must be df with columns "word" and "rating"'
        self.lemmatizer = pymorphy2.MorphAnalyzer()
        self.opinion_words=opinion_words
        self.dataset = None
        self.processed_ds = None

    def run( 
        self,
        filepath_from:str,
        filepath_to:str,
        save=True
    ):
        self.dataset = self.read_data(filepath_from)
        
        syn_net_ds = []
        
        for doc in tqdm(self.dataset):
            doc_id = doc['document_id']
            topic = max(doc['topics_result']['topics'], key=lambda x: x['probability'])
            for t in topic:
                return topic[t]['topic_label']
                topic[t]['topic_label'] = re.sub('\xa0', ' ', topic[t]['topic_label'])
                
            sentences = doc['ner_result']['sentences']
            
            self.syntaxnet_preprocess(sentences, path_to='sy_net_tmp.txt')
            self.run_syntaxnet('sy_net_tmp.txt', 'sy_net_tmp.conll')
            processed_sentences = self.get_processed_sentences('sy_net_tmp.conll', doc_id)

            syn_net_ds.append(
                {
                    'doc_id': doc_id,
                    'topic': topic,
                    'sentences': processed_sentences
                }
            )  
        self.processed_ds = syn_net_ds
        
    
    def read_data(self, file_path:str, test=False):
        '''
        Data in format like "./data/test/input.json"
        Specific for task and so sensetive for changes in input format.
        '''  
        with open(file_path, "r") as read_file:
            dataset = json.load(read_file)
        return dataset
    
    @staticmethod
    def change_text(text:str):
        text = re.sub('\xa0', ' ', text)
        text = re.sub(r'([.,!?()])', r' \1 ', text)
        text = re.sub('  ',' ',text)
        text = re.sub('«', '', text)
        text = re.sub('»', '', text)
        text = re.sub('"', '', text)
        text = re.sub('-', '', text)
        return text.replace(r'. ', '.\n')
    
    def syntaxnet_preprocess(
        self, 
        sentences:list, 
        path_to:str='sy_net_tmp.txt'
    ):        
        with open(path_to, "w") as write_file:
            write_file.writelines(''.join(map(self.change_text, sentences)))
            
    def run_syntaxnet(self, textfile, conllfile):
        command = "cat " + textfile + " | docker run --rm -i inemo/syntaxnet_rus > " + conllfile
        os.system(command)
        
    def get_processed_sentences(self, conll_file:str, doc_id:int):
        processed_sentences = []
        sentence = []
        i = 0
        for line in codecs.open(conll_file, 'r', 'utf-8'):
            if len(line) == 1:
                processed_sentences.append(sentence)
                sentence = []
                i += 1
            else:
                word = line.split("\t")
                sn_word = {
                    'word_id': word[0],
                    'word': word[1],
                    'parent_id': word[6],
                    'tag': word[3],
                    'dependency': word[7],
                    'lemmatized': self.lemmatize(word[1]),
                    'sentence_id': i,
                    'doc_id': doc_id,
                    'pol': self.polarity(self.lemmatize(word[1]))
                }
                sentence.append(sn_word)
        return processed_sentences
    
        
    def lemmatize(self, word:str):
        return self.lemmatizer.parse(word)[0].normal_form.strip()
    
    def polarity(self, word):
        tmp = self.opinion_words.query(f'word == "{word}"')
        if tmp.shape[0] != 0:
            return tmp['rating'].values[0]
        else:
            return 0
        
        
#     def syntaxnet_opinion_without_tw(
#         doc_sn, 
#         doc_id=None,
#         to_file=False, 
#         vw_file=None, 
#     ):
#         i=0

#         posw=Counter()
#         negw=Counter()
#         posw_pair = Counter()
#         negw_pair = Counter()

#         for j in range(doc_sn.shape[0]):
#             row = doc_sn[j:j+1]
#             word = row['lemmatized'].values[0]
#             pol = row['pol'].values[0]


#             if pol!=0:
#                 s_id=row['sentence_id'].values[0]
#                 childs = doc[(doc['sentence_id']==s_id) & (doc['parent_id'] == row['word_id'].values[0])]
#                 #Проверка на наличие отрицаний
#                 neg = childs[childs['dependency']=='neg']
#                 if len(neg)!=0:
#                     print(doc_id, ' не ', word)
#                     pol = -1 if pol==1 else 1
#                 if pol < 0:
#                     negw += Counter({word: abs(pol)})
#                 else:
#                     posw += Counter({word: pol})

#                 #проверка родителя            
#                 p_id=row['parent_id'].values[0]
#                 parent_row = doc[(doc['sentence_id']==s_id) & (doc['word_id']==p_id)]
#                 if len(parent_row)!=0:
#                     parent = parent_row['lemmatized'].values[0]
#                     if pol < 0:
#                         negw += Counter({parent: abs(pol)})
#                         negw_pair += Counter({parent + '_' + word: abs(pol)})

#                     else:
#                         posw += Counter({parent: pol})
#                         posw_pair += Counter({parent + '_' + word: pol})

#                 #глагол (нет проверки на тематичность. нужна ли?)
#                 if row['tag'].values[0] == 'VERB':
#                     #ищем obj и subj
#                     childs = doc[(doc['sentence_id']==s_id) & (doc['parent_id'] == row['word_id'].values[0])]
#                     obj = childs[childs['dependency'].str.contains('obj')]
#                     subj = childs[childs['dependency'].str.contains('subj')]


#                     if len(obj)!=0 and len(subj)!=0: #есть и объект и субъект
#                         if pol < 0:
#                             negw_pair += Counter({subj['lemmatized'].values[0] + '_' + word + '_' + obj['lemmatized'].values[0]: abs(pol)})
#                         else:
#                             posw_pair += Counter({subj['lemmatized'].values[0] + '_' + word + '_' + obj['lemmatized'].values[0]: pol})
#                         print(doc_id,' ',subj['lemmatized'].values[0] + '_' + word + '_' + obj['lemmatized'].values[0])
#                     if len(subj)!=0:

#                         if pol < 0:
#                             negw_pair += Counter({subj['lemmatized'].values[0] + '_' + word: abs(pol)})

#                         else:
#                             posw_pair += Counter({subj['lemmatized'].values[0] + '_' + word: pol})
#                         print(doc_id,' ', subj['lemmatized'].values[0] + '_' + word)
#                     if len(obj)!=0: 

#                         if pol < 0:
#                             negw_pair += Counter({word + '_' + obj['lemmatized'].values[0]: abs(pol)})

#                         else:
#                             posw_pair += Counter({word + '_' + obj['lemmatized'].values[0]: pol})
#                         print(doc_id,' ',word + '_' + obj['lemmatized'].values[0])
#                 #advmod
#         if to_file==True:
#             doc_info = u''
#             doc_info=doc_info+u"|neg_pol "

#             for w in negw:
#                 doc_info=doc_info+u" "+w+u":"+str(negw[w])
#             for w in negw_pair:
#                  doc_info=doc_info+u" "+w+u":"+str(negw_pair[w])

#             doc_info = doc_info + u" |pos_pol "
#             for w in posw:
#                 doc_info=doc_info+u" "+w+u":"+str(posw[w])        
#             for w in posw_pair:
#                  doc_info=doc_info+u" "+w+u":"+str(posw_pair[w])
#             doc_info+="\n"                

#             return doc_info