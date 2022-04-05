from nltk import DependencyGraph
import codecs
import itertools
import numpy as np
import pandas as pd
import re
import os
import pymorphy2
import math
from collections import Counter
from stop_words import get_stop_words
import codecs
import os.path
from tqdm import tqdm

stop_words = get_stop_words('russian')
lemmatizer = pymorphy2.MorphAnalyzer()

# Transforms conll into lists:
def get_lists(sent_dep):
    dependencies = list(map(lambda x: int(x['parent_id'])-1, sent_dep))
    pos = list(map(lambda x: x['tag'], sent_dep))
    tp = list(map(lambda x: x['dependency'], sent_dep))
    words = list(map(lambda x: x['lemmatized'], sent_dep))
    
    for i in range(len(tp)):
        # Find 'and' sequences
        if tp[i] == 'conj' and pos[i] == 'VERB':
            ids = [x for x in range(len(tp)) if dependencies[x] == dependencies[i] and tp[x] == 'nsubj'] 
            for j in ids:
                words.append(words[j])
                pos.append(pos[j])
                tp.append(tp[j])
                dependencies.append(i)
        elif tp[i] == 'conj' and pos[i] != 'VERB':
            dep = dependencies[i]
            pos[i] = pos[dep]
            dependencies[i] = dependencies[dep]
            tp[i] = tp[dep]
            
        # Find complex verbs
        if tp[i] in ['xcomp','dep']:
            dep = dependencies[i]
            words[dep] = words[dep] + ' ' + words[i]
            ids = [x for x in range(len(tp)) if dependencies[x] == i]
            for j in ids:
                dependencies[j] = dep
            pos[dep] = u'VERB'
            pos[i] = 'ADD_VERB'
            tp[i] = 'ADD_VERB'
            
        # Adjective triplets
        if tp[i] == 'ADJ' and pos[dependencies[i]] == 'VERB':
            dep = dependencies[i]
            words[dep] = words[dep]+' '+words[i]
        
        # Determine negative verbs
        if tp[i] == u'neg':
            dep = dependencies[i]
            words[dep] = words[i]+' '+words[dep]
        
        # Substitude words with their names if present
        if tp[i] == u'name':
            dep = dependencies[i]
            words[dep] = words[i]

    return words, pos, dependencies, tp 
                
# Find triplets in conll processed form        
def get_triplets(processed_sentence):
    words, pos, dependencies, tp = get_lists(processed_sentence)
    
    ids = range(len(words))
    
    # regular triplets
    verbs = [x for x in ids if pos[x] == u'VERB' and tp[x] != 'amod']
    for i in verbs:
        verb_subjects = [words[x] for x in ids if tp[x] in ['nsubj','nsubjpass'] and dependencies[x] == i]
        if len(verb_subjects) == 0:
            verb_subjects.append(u'imp')
        verb_objects = [words[x] for x in ids if tp[x] == 'dobj' and dependencies[x] == i]
        if len(verb_objects) == 0:
            verb_objects.append(u'imp')
        for subj, obj in itertools.product(verb_subjects, verb_objects):
            triplets.append([subj, words[i], obj])
       
    # participle triplets
    participles = [x for x in ids if pos[x] == u'VERB' and tp[x] == 'amod']
    for i in participles:
        participle_subjects = [words[x] for x in ids if dependencies[i] == x]
        if len(participle_subjects) == 0:
            participle_subjects.append(u'imp')
        participle_objects = [words[x] for x in ids if tp[x] == 'dobj' and dependencies[x] == i]
        if len(participle_objects) == 0:
            participle_objects.append(u'imp')
        for subj, obj in itertools.product(participle_subjects, participle_objects):
            triplets.append([subj, words[i], obj])
            
    # implicit noun-noun triplets
    appos = [x for x in ids if tp[x] == u'appos']
    for i in appos:
        obj = words[dependencies[i]]
        triplets.append([words[i], u'есть', obj])

                
    #adjectives triplets
    adjectives = [x for x in ids if pos[x] == 'ADJ' and tp[x] == 'amod']
    for adj in adjectives:
        triplets.append([words[dependencies[adj]], u'есть', words[adj]])
    return triplets


# Get triplets from text doc or conll doc    
def get_doc_triplets(doc):
    processed_sentences = doc['conll']
    text_triplets = []
    for sent in tqdm(processed_sentences):
        text_triplets.extend(get_triplets(sent))
    return text_triplets

# Extract all subjects from triplet list
def subjects_from_triplets(triplet_list):
    return [x[0] for x in triplet_list if x[0] != u'imp' and x[0] not in stop_words]


# Extract all objects from triplet list
def objects_from_triplets(triplet_list):
    return [x[2] for x in triplet_list if x[2] != u'imp' and x[2] not in stop_words]


def get_subjects_from_triplet_lists(triplet_lists):
    subject_lists = []
    for triplets in triplet_lists:
        subject_lists.append(subjects_from_triplets(triplets))
    return subject_lists
        
def prepare_spo(doc):
    triplets = get_doc_triplets(doc)
    subjects = subjects_from_triplets(triplets)
    objects = objects_from_triplets(triplets)
            
    subj_res = []
    obj_res = []
    
    for subject in set(subjects):
        if subject == u'—':
            continue
        print(subject)
        subject = re.sub(':', '', subject)
        subj_res.append(subject.lower()+':'+str(subjects.count(subject)))
    
    for obj in set(objects):
        if obj == u'—':
            continue
        obj = re.sub(':', '', obj)
        obj_res.append(obj.lower()+':'+str(objects.count(obj)))
     
    return subj_res, obj_res