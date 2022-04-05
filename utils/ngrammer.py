import pickle
from itertools import groupby, product
from rusenttokenize import ru_sent_tokenize
from conllu import parse
from pymorphy2 import MorphAnalyzer
import re
from deeppavlov import build_model, configs
from deeppavlov.core.common.file import read_yaml, read_json
from ufal.udpipe import Model, Pipeline

from stop_words import get_stop_words
from isanlp_srl_framebank.pipeline_default import PipelineDefault 

udpip_model = Model.load('../udpipe_syntagrus.model')
pipeline = Pipeline(udpip_model, 'generic_tokenizer', '','','')
# syntax_parser = read_json("syntax_parser_vx.json")
# model = build_model(syntax_parser)
morph = MorphAnalyzer()

class Extractor():
    def __init__(self):
        self.text = ''
        self.s_idx = -1
        self.noun_trash = ['г.', '.', 'имя']
        self.morph_replace = {'им':'им'}
        
        self.roles_map = {'pred': 'pred',
         'агенс': 'agent',
         'адресат': 'goal',
         'говорящий': 'speaker',
         'исходный посессор': 'source',
         'каузатор': 'source',
         'конечная точка': 'goal',
         'конечный посессор': 'goal',
         'контрагент': 'instrument',
         'контрагент социального отношения': 'instrument',
         'место': 'locative',
         'начальная точка': 'source',
         'пациенс': 'patient',
         'пациенс перемещения': 'patient',
         'пациенс социального отношения': 'patient',
         'потенциальная угроза': 'locative',
         'потенциальный пациенс': 'patient',
         'предмет высказывания': 'instrument',
         'предмет мысли': 'experiencer',
         'признак': 'theme',
         'признак действия': 'theme',
         'причина': 'source',
         'результат': 'goal',
         'ситуация в фокусе': 'theme',
         'содержание высказывания': 'instrument',
         'содержание действия': 'instrument',
         'содержание мысли': 'instrument',
         'способ': 'instrument',
         'срок': 'locative',
         'статус': 'locative',
         'стимул': 'source',
         'субъект восприятия': 'experiencer',
         'субъект ментального состояния': 'experiencer',
         'субъект перемещения': 'experiencer',
         'субъект поведения': 'experiencer',
         'субъект психологического состояния': 'experiencer',
         'субъект социального отношения': 'experiencer',
         'сфера': 'theme',
         'тема': 'theme',
         'траектория': 'theme',
         'цель': 'goal',
         'эффектор': 'agent',
         'эталон': 'source'}
        self.stop_words = get_stop_words('russian') if stop_words is None else stop_words
        self.ppl = PipelineDefault(
            address_morph=('localhost', 3333),
            address_syntax=('localhost', 3334),
            address_srl=('localhost', 3335)
        )

    def to_morph(self, tag):
        if tag == 'Fem': return 'femn'
        if tag == 'Masc': return 'masc'
        if tag == 'Neut': return 'neut'

    def to_conll(self, sent):
        sent = '\n'.join([line for line in sent.split('\n') if not line.startswith('#')])
        sent = sent.replace('\troot\t', '\tROOT\t')
        return sent

    def norm_word(self, token, head=None):
        morph_token = morph.parse(str(token))[0]
        if head:
            try:
                try:
                    morph_head = self.to_morph(head['feats']['Gender'])
                except:
                    morph_head = morph.parse(str(head))[0].tag.gender
                norm = morph_token.inflect({'sing', 'nomn', morph_head}).word
            except:
                norm = morph_token.normal_form
        else:
            if token['form'] in self.morph_replace.keys():
                norm = self.morph_replace[token['form']]
            else:
                try:
                    norm = token['lemma']
                except:
                    norm = morph_token.normal_form
        return norm, morph_token


    def postprocessing(self, string):
        string = string.replace(' .', '.')\
                    .replace(' " "', '') \
                    .replace(' « »', '')
        return string

    def fill_token(self, token, np):
        #TODO проверить что найденное вхождение в заданном предложении
        start_idx = self.text.find(token['form'], self.s_idx)
        end_idx = start_idx + len(token['form']) - 1
        token['start_idx'] = start_idx
        token['end_idx'] = end_idx
        token['np'] = np
        return token


    #Возврощаем все amod для найденной head
    def extract_amod(self, conll_form, nps):
        for obj in parse(conll_form)[0].filter(deprel='amod'):
            filter_head = parse(conll_form)[0].filter(id=obj['head'])
            if filter_head:
                head = filter_head[0]
                chain = []
                children = parse(conll_form)[0].filter(head=obj['head'], deprel='amod')
                for token in children:
                    norm_token, morph_token = self.norm_word(token, head)
                    chain.append([norm_token, token])
                chain.append([head['lemma'], head])
                key = self.postprocessing(' '.join([w[0] for w in chain]).lower())
                if len(chain) > 1 and key not in nps.keys():
                    chain = [self.fill_token(token[1], token[0]) for token in chain]
                    key = re.sub('^[\W\s]*|[\W\s]*$','', key)
                    nps[key] = chain
        return nps

    #Возврощаем все amod а так же следующий за ним элемент (любой) а далее рекурсивно собирает nmod
    def extract_amod_nmod(self, conll_form, nps):
        def check_link(id, conll_form, chain):
            for nmod in parse(conll_form)[0].filter(deprel = 'nmod'):
                if nmod['head'] == id and abs(nmod['id']-id)==1:
                    chain.append([nmod['form'], nmod])
                    check_link(nmod['id'], conll_form, chain)
            return chain
        for amod in parse(conll_form)[0].filter(deprel='amod'):
            chain = []
            filter_head = parse(conll_form)[0].filter(id=amod['head'])
            if filter_head:
                head = filter_head[0]
                norm_amod = self.norm_word(amod['form'], head)
                chain.append([norm_amod[0], amod])
                chain.append([head['lemma'], head])
                chain = check_link(head['id'], conll_form, chain)
                key = self.postprocessing(' '.join([w[0] for w in chain]).lower())
                if len(chain) > 1 and key not in nps.keys():
                    chain = [self.fill_token(token[1], token[0]) for token in chain]
                    key = re.sub('^[\W\s]*|[\W\s]*$','', key)
                    nps[key] = chain
        return nps

    #Разбераються случаи когда в тексте есть nsubj и череда nmod после него
    def extract_nsubj_nmod(self, conll_form, nps):
        def check_link(id, conll_form, chain):
            for nmod in parse(conll_form)[0].filter(deprel = 'nmod'):
                if nmod['head'] == id:
                    chain.append([nmod['form'], nmod])
                    check_link(nmod['id'], conll_form, chain)
            return chain
        for nsubj in parse(conll_form)[0].filter(deprel = 'nsubj'):
            chain = []
            norm_nsubj, morph_nsubj = self.norm_word(nsubj)
            chain.append([norm_nsubj, nsubj])
            chain = check_link(nsubj['id'], conll_form, chain)
            key = self.postprocessing(' '.join([w[0] for w in chain]).lower())
            if len(chain) > 1 and key not in nps.keys():
                chain = [self.fill_token(token[1], token[0]) for token in chain]
                key = re.sub('^[\W\s]*|[\W\s]*$','', key)
                nps[key] = chain
        return nps

    #Находит последовательность nmod (допускаються amod в цепочке)
    #TODO имеет место сомнительные вхождения - посмотреть перекрытие или изменения
    def extract_nmod(self, conll_form, nps):
        def check_link(id, conll_form, chain):
            for nmod in parse(conll_form)[0].filter(id=id+1):
                if nmod['deprel']=='nmod' or nmod['deprel']=='amod':
                    chain.append([nmod['form'], nmod])
                    check_link(nmod['id'], conll_form, chain)
            return chain
        for nmod in parse(conll_form)[0].filter(deprel='nmod'):
            chain = []
            norm_nmod, morph_nmod = self.norm_word(nmod)
            chain.append([norm_nmod, nmod])
            chain = check_link(nmod['id'], conll_form, chain)
            key = self.postprocessing(' '.join([w[0] for w in chain]).lower())
            if len(chain) > 1 and len(chain) < 4  and key not in nps.keys():
                chain = [self.fill_token(token[1], token[0]) for token in chain]
                key = re.sub('^[\W\s]*|[\W\s]*$','', key)
                nps[key] = chain
        return nps

    def extract_nmod_appos(self, conll_form, nps):
        for noun in parse(conll_form)[0].filter(deprel='nmod', upos='NOUN'):
            chain = []
            for appos in  parse(conll_form)[0].filter(deprel='appos', head=noun['id']):
                chain.append([noun['lemma'], noun])
                chain.append([appos['form'], appos])
            key = self.postprocessing(' '.join([w[0] for w in chain]).lower())
            if len(chain) > 1 and key not in nps.keys():
                chain = [self.fill_token(token[1], token[0]) for token in chain]
                key = re.sub('^[\W\s]*|[\W\s]*$','', key)
                nps[key] = chain
        return nps

    def extract_obl_nmod(self, conll_form, nps):
        def check_link(id, conll_form, chain):
            for nmod in parse(conll_form)[0].filter(deprel = lambda x: x=='nmod' or x == 'iobj'):
                if nmod['head'] == id and abs(nmod['id']-id)==1:
                    chain.append([nmod['form'], nmod])
                    check_link(nmod['id'], conll_form, chain)
            return chain
        for obl in parse(conll_form)[0].filter(deprel='obl', upos='NOUN'):
            chain = []
            try:
                for amod in parse(conll_form)[0].filter(deprel='amod', head=obl['id']):
                    norm_amod = self.norm_word(amod['form'], obl)
                    chain.append([norm_amod[0], amod])
            except:
                pass
            chain.append([obl['lemma'], obl])
            chain = check_link(obl['id'], conll_form, chain)
            key = self.postprocessing(' '.join([w[0] for w in chain]).lower())
            if len(chain) > 1 and key not in nps.keys():
                chain = [self.fill_token(token[1], token[0]) for token in chain]
                key = re.sub('^[\W\s]*|[\W\s]*$','', key)
                nps[key] = chain
        return nps

    def extract_foreign_name(self, conll_form, nps):
        def get_head(token):
            return token['head']
        sorted_chain = sorted(parse(conll_form)[0].filter(deprel='flat:foreign', upos='PROPN'), key=get_head)
        grouped_chain = [list(it) for k, it in groupby(sorted_chain, get_head)]
        if len(grouped_chain) > 0:
            for group in grouped_chain:
                key = self.postprocessing(' '.join([w['form'] for w in group]).lower())
                chain = [self.fill_token(token, token['form'].lower()) for token in group]
                key = re.sub('^[\W\s]*|[\W\s]*$','', key)
                nps[key] = chain
        return nps

    def extract_noun(self, conll_form, nps):
        #TODO придумать фильтры: очень папулярные, только связанные или не связанные,
        for noun in parse(conll_form)[0].filter(upos='NOUN'):
            if noun['deprel'] in ['nmod', 'root', 'obl', 'obj', 'nsubj','appos']:
                key = noun['lemma'].lower()
                if key not in nps.keys() and key not in self.noun_trash:
                    key = re.sub('^[\W\s]*|[\W\s]*$','', key)
                    nps[key] = self.fill_token(noun, key)
        return nps

    def get_lists(self, conll_form:'str[conll]'):
        '''Transforms conll into lists:'''
        parsed_conll = parse(conll_form)[0]
        
        dependencies = list(map(lambda x: int(x['head'])-1, sent_dep))
        pos = list(map(lambda x: x['upos'], sent_dep))
        tp = list(map(lambda x: x['deprel'], sent_dep))
        words = list(map(lambda x: self.norm_word(x['form']), sent_dep))

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
            if pos[i] == 'ADJ' and pos[dependencies[i]] == 'VERB':
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

    def extract_triplets(self, conll_form:'str[conll]'):
        '''Find triplets in conll processed form'''
        words, pos, dependencies, tp = self.get_lists(conll_form)
        
        triplets = []
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
            for subj, obj in product(verb_subjects, verb_objects):
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
    
    def extract_fillmore(self, text:str):
        def get_roles(text:str):
            res = self.ppl(text)
            data = []
            for sent_num, ann_sent in enumerate(res['srl']):
                for i, event in enumerate(ann_sent):
                    sr = {'lemma': res['lemma'][sent_num][event.pred[0]], 'role': 'pred'}
                    data.append(sr)
                    for arg in event.args:
                        sr = {'lemma': res['lemma'][sent_num][arg.begin], 'role': arg.tag}
                        data.append(sr)
            return data
            
        def change_text(text:str):
            text = re.sub('\xa0', ' ', text)
            text = re.sub(r'([.,!?()])', r' \1 ', text)
            text = re.sub('  ',' ',text)
            text = re.sub('«', '', text)
            text = re.sub('»', '', text)
            text = re.sub('"', '', text)
            text = re.sub('-', '', text)
            return text.replace(r'. ', '.\n')
        
        raw_roles = get_roles(change_text(text))
        for word in raw_roles:
            word['role'] = self.roles_map[word['role']]
            
        pairs = dict(Counter(pair['lemma']+'--'+pair['role'] for pair in tmp))
        return pairs

    
    def extract_nps(self, conll_form, text, s_idx, nps=None):
        self.text = text
        self.s_idx = s_idx
        if nps == None:
            nps = {}
        nps = self.extract_amod_nmod(conll_form, nps)
        nps = self.extract_amod(conll_form, nps)
        nps = self.extract_nsubj_nmod(conll_form, nps)
        nps = self.extract_nmod_appos(conll_form, nps)
        nps = self.extract_nmod(conll_form, nps)
        nps = self.extract_obl_nmod(conll_form, nps)
        nps = self.extract_foreign_name(conll_form, nps)
        nps = self.extract_noun(conll_form, nps)
        return nps


    def batch_nps_extraction(self, batch:list):
        result = []
        for text in batch:

            text_result = []
            sents = ru_sent_tokenize(text)

            syntax_parser_output = [self.to_conll(pipeline.process(sent)) for sent in sents]
            
            #syntax_parser_output = model(sents)
            for conll_form, sent in zip(syntax_parser_output, sents):
                nps = {}
                triplets=[]
                s_idx = text.find(sent)
                if sent and conll_form:
                    nps = self.extract_nps(conll_form, text, s_idx, nps)
                    triplets = self.extract_triplets(conll_form)
                text_result.append({
                    'begin_sent': s_idx, 
                    'end_sent': s_idx + len(sent), 
                    'nps': nps,
                    'triplets':triplets
                })
            text_result.append({'fillmore': self.extract_fillmore(text)})
            result.append([text_result])
        return result


Extractor().batch_nps_extraction(["В 1698 году в Бастилию доставили заключенного, лицо которого скрывала ужасная железная маска. Его имя было неизвестно, а в тюрьме он числился под номером 64489001. Созданный ореол таинственности породил немало версий, кем мог быть этот человек в маске. Узник в железной маске на анонимной гравюре времён Французской революции (1789). О переведенном из другой тюрьмы узнике начальство не знало ровным счетом ничего. Им было предписано поместить человека в маске в самую глухую камеру и не разговаривать с ним. Через 5 лет заключенный умер. Он был похоронен под именем Марчиалли. Все вещи покойника сожгли, а стены расковыряли, чтобы не осталось каких-либо записок. Когда в конце XVIII века под натиском Великой французской революции Бастилия пала, новое правительство обнародовало документы, проливающие свет на судьбы заключенных. Но в них о человеке в маске не было ни единого слова. Бастилия – французская тюрьма. Иезуит Гриффе, бывший духовником в Бастилии в конце XVII века, писал, что в тюрьму доставили узника в бархатной (не в железной) маске. К тому же, заключенный надевал ее только тогда, когда в камере кто-нибудь появлялся. С медицинской точки зрения, если бы узник действительно носил личину из металла, то это неизменно изуродовало его лицо. Железной маску «сделали» литераторы, которые делились своими предположениями, кем на самом деле мог быть этот загадочный заключенный. Человек в железной маске. Впервые об узнике в маске упоминается в «Тайных записках Персидского двора», изданных в 1745 году в Амстердаме. Согласно «Запискам», заключенным № 64489001 был не кто иной, как внебрачный сын Людовика XIV и его фаворитки Луизы Француазы де Лавальер. Он носил титул герцога Вермандуа, якобы дал пощечину своему брату Великому Дофину, за что и угодил за решетку. На самом деле, эта версия неправдоподобна, т. к. внебрачный сын французского короля умер в 16-летнем возрасте в 1683 году. А по записям духовника Бастилии иезуита Гриффе, неизвестного заключили в тюрьму в 1698 году, а скончался он в 1703 году. Кадр из к/ф «Человек в Железной маске» (1998). Франсуа Вольтер в своем произведении «Век Людовика XIV», написанном в 1751 году, впервые указал, что Железной Маской вполне мог быть брат-близнец Короля-солнце. Чтобы не возникло проблем с престолонаследием, одного из мальчиков воспитывали тайно. Когда же Людовик XIV узнал о существовании брата, то обрек его на вечное заточение. Эта гипотеза настолько логично объясняла наличие у заключенного маски, что она стала самой популярной среди прочих версий и впоследствии была не раз экранизирована режиссерами. Под маской мог скрываться итальянский авантюрист Эрколь Антонио Маттиоли. Существует мнение, что маску вынужден был носить известный итальянский авантюрист Эрколь Антонио Маттиоли. Итальянец в 1678 году заключил соглашение с Людовиком XIV, по которому обязывался заставить своего герцога сдать крепость Казале королю в обмен за вознаграждение в 10 000 скудо. Деньги авантюрист взял, но договор не выполнил. Более того, Маттиоли выдал этот государственный секрет еще нескольким странам за отдельное вознаграждение. За эту измену французское правительство отправило его в Бастилию, заставив надеть маску. Российский император Петр I. Некоторые исследователи выдвигали совсем уж неправдоподобные версии о человеке в железной маске. По одной из них, этим узником мог быть российский император Петр I. Именно в тот период Петр I находился в Европе со своей дипломатической миссией («Великое посольство»). Самодержца якобы заточили в Бастилию, а вместо него домой отправили подставное лицо. Мол, как иначе объяснить тот факт, что царь уезжал из России свято чтившим традиции христианином, а обратно вернулся типичным европейцем, пожелавшим сломать патриархальные устои Руси. Если вам понравился пост, пожалуйста, поделитесь ими со своими друзьями!"])
