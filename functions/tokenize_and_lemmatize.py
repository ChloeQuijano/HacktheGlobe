from nltk.corpus import wordnet
import nltk
nltk.download('averaged_perceptron_tagger')
import datetime
import logging
from nltk.corpus import wordnet
from .get_wordnet_pos import *
import sys
sys.path.append("../wordninja")
from wordninja import wordninja as wnj
import csv
import os

def tokenize_and_lemmatize(params):
    """
    This function parses the text of input files and pre-processes them to produce cleaned, lemmatized token lists of each document.
    
    @param: text_lists:  The 'text' content that needs to be tokenized by the logic of this function.
    @param: lemmatizer:  NLTK object to lemmatize tokens
    @param: punctuations: String of all punctuation to be removed from tokens during pre-processing
    @param: stop_words: The list of stop words from the spacy "en" language model
    @param: exclusion_list: A list of words to be excluded from the vocabulary
    @param: inclusion_list: A list of words to be included in the vocabulary
    @param: call_id: The unique id of the set of parameters to pass to this function
    @param: punctuations: string of punctuation marks and/or symbols to be excluded as tokens
    @param: output_dir: A log of all words removed from the corpus (stop words, etc.) will be written to this directory. 
                        If empty, no log file will be written.
    @return value: status: True for success or False for Failure
                   token_list: In case the execution is successful, a list of tokens that are extracted from 'text'
                   msg: In case of error, a message summarise the error
    """


    ## Status is failure as a default ++
    status   = {
        'status'      : False
        , 'msg'       : 'Default error message'
        , 'token_list': {}
    }
    ## Status is failure as a default --

    token_lists = {}
    text_lists   = params.get('text_lists', {})
    lemmatizer = params.get('lemmatizer', None)
    stop_words = params.get('stop_words',[])
    punctuations = params.get('punctuations','')
    output_dir = params.get('output_dir',('./','log_file'))
    exclusion_list = params.get('exclusion_list',[])
    inclusion_list = params.get('inclusion_list',[])
    
    ## Tokens processed will be written out to a log under the output_dir in CSV format
    log_file = 'token_transformation_'
    
    ## Error handling ++
    if {} == text_lists:
        status['status'] = False
        status['msg']    = 'Lists of text are empty. Nothing to tokenize.'
        return status

    if None == lemmatizer:
        status['status'] = False
        status['msg']    = 'No lemmatizer object was provided. This is required for tokenization.'
        return status
    
    ## Error handling --
    
    ## Business logic ++        
    inclusion_list_lower=[il.lower() for il in inclusion_list]
    
    ## Replace these characters prior to wordninja splitting on them
    _tbl = str.maketrans(
        {
            '-': None,
            '~': None,
            '/': ' ',
            '\\': ' ',
            ':': ' ',
            ';': ' ',
            '.': None,
            ',': None,
            '\t': None,
            '\r': None,
            '\n': None,
        }
    )
    
    ## Modify stop word list to remove words in the inclusion list, and add words in the exclusion list
    [stop_words.add(el.lower()) for el in exclusion_list]
    [stop_words.discard(il) for il in inclusion_list_lower]
    
    out_log=[]
    token_lists = {}
    
    if output_dir:
        wfile = open(os.path.join(output_dir,log_file+'.log'), 'wt')

    for file in text_lists:
        token_lists[file]=[]
        for string in text_lists[file]:
            doc_tokens=[]
            ## Remove punctuations such as '~' and '-' to join words
            string = string.translate(_tbl)
            string_words = wnj.split(string)
            string_pos = nltk.pos_tag(string_words)
            for word, pos in string_pos:
                token = word.lower()
                ## Change to lowercase
                if token in inclusion_list_lower:
                    doc_tokens.append(token)
                    continue
                ## Exlude tokens that are punctuations
                elif token in punctuations:
                    out_log.append(token)
                    continue
                ## Exclude stop words
                elif token in stop_words:
                    out_log.append(token)
                    continue
                else:
                    ## Tag words with their part of speech and lemmatize
                    wn_pos = get_wordnet_pos(pos)['wordnet_tag']
                    if wn_pos == '':
                        doc_tokens.append(lemmatizer.lemmatize(token))
                    else:
                        doc_tokens.append(lemmatizer.lemmatize(token,wn_pos))
                        
                if output_dir:
                    wfile.write('{},{},{}\n'.format(file, word, token))
                    
            token_lists[file].append(doc_tokens)
            
    ## Write excluded tokens to a log file
    if output_dir:
        wfile.close()
        logger = logging.getLogger('tokenize_and_lemmatize')
        logger.setLevel(logging.DEBUG)
        # create file handler 
        fh = logging.FileHandler(os.path.join(output_dir,'output_log')+'.log')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.debug("Words removed during pre-processing: {}".format(', '.join(map(str, list(set(out_log)))))+'\n')
        
        
    ## Business logic --
    
    ## Return for success ++
    status['status']     = True
    status['msg']        = 'Text tokenized.'
    status['token_list'] = token_lists
    ## Return for success --

    return status
