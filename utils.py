import numpy as np
import os
import tarfile
import pickle
import sys
from datetime import date
from xml.dom import minidom
from sklearn.feature_extraction.text import CountVectorizer

import nltk
import nltk.data
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

nltk.download('state_union')
nltk.download('wordnet')
nltk.download('punkt')

#train_text = state_union.raw("2005-GWBush.txt")
custom_sent_tokenizer = RegexpTokenizer(r'\w+') #PunktSentenceTokenizer(train_text)

def process_content(tokenized):
    try:
        words_with_POS = []
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            words_with_POS.extend(tagged)
            #print(tagged)
        #print(words_with_POS)
        return words_with_POS

    except Exception as e:
        print(str(e))
        
def POS_tag(text):
    tokenized = custom_sent_tokenizer.tokenize(text.lower())
    stops = stopwords.words('english')
    tokenized = [x for x in tokenized if x not in stops and x.isalpha()]
    return tokenized

def lemmatize(words_with_POS):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    #words_in_text = re.split('; |, | |\n|\. |', text) #this may still leave in other punctuation marks
    #print(words_in_text)
    for word_with_POS in words_with_POS:
        word = word_with_POS[0]
        POS = word_with_POS[1]
        if POS[0] == 'N':
            lemmatized_tokens.append(lemmatizer.lemmatize(word,pos=wordnet.NOUN))
        elif POS[0] == 'J':
            lemmatized_tokens.append(lemmatizer.lemmatize(word,pos=wordnet.ADJ))
        elif POS[0] == 'V':
            lemmatized_tokens.append(lemmatizer.lemmatize(word,pos=wordnet.VERB))
        elif POS[0] == 'R':
            lemmatized_tokens.append(lemmatizer.lemmatize(word,pos=wordnet.ADV))
        else:
            lemmatized_tokens.append(lemmatizer.lemmatize(word))
        
    return lemmatized_tokens

def extract_corpus(folder, extract=True):
    file_list = [];
    if extract:
        for file in os.listdir(folder):
            if file.endswith(".tgz"):
                file_list.append(os.path.join(folder, file))

            print 'extracting', file_list[-1]
            tar = tarfile.open(file_list[-1]);
            tar.extractall(path=folder);        
            tar.close();
        
    articleList = [];
    n = 0;
    badFiles = [];
    for root, dirs, files in os.walk(folder):
        for name in files:
            if name.endswith(".xml"):
                A = NewsArticle(os.path.join(root, name))
                if A.parse():
                    A.add_lemmas()
                    articleList.append(A);
                    n += 1;
                    if np.mod(n, 100) == 0:
                        print '%d: %s: Headline = %s, %s' % (n, A.fileName, A.headline, A.date.strftime("%d/%m/%y"))
                else:
                    badFiles.append(A.fileName);
                    
                    
    print 'Processed %d articles' % len(articleList)
    if len(badFiles) > 0:
        print "Some files could not be processed"
        print badFiles
        
    return articleList
    
def save_data(fileName, data):
    with open(fileName + '.pickle', 'w') as f:
        pickle.dump(data, f);

def load_data(fileName):
    with open(fileName + '.pickle', 'r') as f:
        data = pickle.load(f);
    return data
 
def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)


class NewsArticle:   
    def __init__(self, fileName):
        self.fileName = fileName;
        self.headline = ''
        self.section = ''
        self.desk = ''
        self.date = date.today();
        self.url = ''
        # self.lead_paragraph = '' # CHANGED: no longer imported
        self.full_text = ''
        self.lemmas = [] #  CHANGED: lemmas[i] is a list of lemmas of the (i-1)th sentence of full_text
        
    def parse(self):
        try:
            xmldoc = minidom.parse(self.fileName)

            # Headline
            itemlist = xmldoc.getElementsByTagName('title') 
            self.headline = itemlist[0].firstChild.nodeValue;

            # Section & date
            itemlist = xmldoc.getElementsByTagName('meta') 
            for item in itemlist:
                if item.attributes['name'].value == "online_sections":
                    self.section = item.attributes['content'].value;
                elif item.attributes['name'].value == "dsk":
                    self.desk = item.attributes['content'].value;
                elif item.attributes['name'].value == "publication_day_of_month":
                    day = item.attributes['content'].value;
                elif item.attributes['name'].value == "publication_month":
                    month = item.attributes['content'].value;
                elif item.attributes['name'].value == "publication_year":
                    year = item.attributes['content'].value;

            self.date = date(int(year), int(month), int(day));

            # URL
            itemlist = xmldoc.getElementsByTagName('pubdata') 
            self.url = itemlist[0].attributes["ex-ref"].value;

            # Text
            itemlist = xmldoc.getElementsByTagName('block') 
            self.lead_paragraph = '';
            self.full_text = '';
            for item in itemlist:
                if item.attributes['class'].value == 'lead_paragraph':
                    continue 
                    # No longer imports lead paragraph
                    #for child in item.childNodes:
                    #    if child.nodeType == child.ELEMENT_NODE:
                    #        self.lead_paragraph += getText(child.childNodes);
                elif item.attributes['class'].value == 'full_text':
                    for child in item.childNodes:
                        if child.nodeType == child.ELEMENT_NODE:
                            self.full_text += getText(child.childNodes);
            return True
        except:
            print "Error (%s) while parsing file = %s" % (sys.exc_info()[0], self.fileName)
            return False;        

    def article_info(self):
        print 'Headline = %s' % (self.headline)
        print 'URL = %s' % (self.url)
        print 'Section = %s | Date = %s' % (self.section, self.date.strftime("%d/%m/%y"))
        print 'Desk = %s' % (self.desk)
        #print '\nLead paragraph: \n%s' % (self.lead_paragraph)
        print '\nFull text: \n%s' % (self.full_text)
        print '\nLemmas: \n%s\n' % (self.lemmas)
 
    def add_lemmas(self):
        # self.lemmas = lemmatize(process_content(POS_tag(self.full_text)))
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.lemmas = [];
        try:
            sentences = tokenizer.tokenize(self.full_text)
            for s in sentences:
                self.lemmas.append(lemmatize(process_content(POS_tag(s))))
        except NameError:
            raise
        except:
            print "Error (%s) while lemmatizing sentences of article entitled %s" % (sys.exc_info()[0], self.headline)
            
            
############################
# co-occurrence functions
############################
def cooccurrence(articleList):
    # Compute co-occurrence matrix from aticles in articleList. 
    # The ij entry of the co-occurrence matrix corresponds to the co-occurrence between words nodeNames[i] and nodeNames[j]
    count_model = CountVectorizer(ngram_range=(1,1)) # default unigram model
    lemmatized_sentences = [];
    for article in articleList:
        for s in article.lemmas:
            lemmatized_sentences.append(' '.join(s))
            
    X = count_model.fit_transform(lemmatized_sentences)
               
    cooccurrenceMatrix = (X.T * X) # this is co-occurrence matrix in sparse csr format
    cooccurrenceMatrix.setdiag(0) # set diagonal to 0 (avoid self loops)
    nodeNames = count_model.get_feature_names();
    
    return cooccurrenceMatrix, nodeNames   