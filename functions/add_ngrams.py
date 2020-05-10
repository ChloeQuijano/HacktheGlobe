from gensim.models.phrases import Phrases,Phraser
import pickle
import os.path
    
def add_ngrams(params):
    """
    This function detects phrases (sequence of two or three words that occur together frequently), concatenates them into n-grams using a 
    delimiter, and adds them as tokens to the document's vocabulary.
    
    @param: output_dir: The directory path where the output of this function (token dictionary) is saved
    @param: lemma_list:  The lemmatized tokens that are used to search for ngrams
    @param: n: Contiguous sequence of "n" tokens are extracted as tokens - 'bigrams' or 'trigrams'
    @param: min_count: The number of instances a potential ngram must appear in the corpus to be considered one.
    @param: delimiter: The delimiting character that is used to combine ngram tokens. (e.g. multi~unit)
    @param: scoring: The statistical scoring function of the association of two tokens
    @param: threshold: The minimum cutoff of the association score for two tokens to be considered an ngram
    @param: call_id: The id to specify the set of input parameters (in param_dict) to be passed to the function
    @return value: status: True for success or False for Failure
                   ngrams_list: In case the execution is successful, a list of lemmatized tokens including ngrams
                   msg: In case of error, a message summarise the error
    """
  
    ## Status is failure as a default ++
    status   = {
        'status'      : False
        , 'msg'       : 'Error'
        , 'ngrams_list': {}
    }
    ## Status is failure as a default --
    token_dictionary_dir = params.get('output_dir', None)
    lemma_list = params.get('lemma_list', {})
    n = params.get('n','trigrams')
    min_count = params.get('min_count',5)
    delimiter = params.get('delimiter',b'~')
    scoring = params.get('scoring','npmi')
    threshold = params.get('threshold',0.9)
    
    ## Error handling ++
    if None == token_dictionary_dir:
        status['status'] = False
        status['msg'] = 'No output directory provided.'
        return status
    
    if {} == lemma_list:
        status['status'] = False
        status['msg']    = 'Lemmatized token list is empty - cannot find ngrams.'
        return status
    ## Error handling --
    
    ## Business Logic ++
    ngrams_list={}
    all_tokens=[]
    for file in lemma_list:
        for doc in lemma_list[file]:
            all_tokens.append(doc)
            
    ## Model bigrams
    bigrams = Phrases(all_tokens, min_count=min_count,delimiter=delimiter,scoring=scoring,threshold=threshold)
    bigrams_phraser = Phraser(bigrams)
    
    ## Model trigrams
    if n == 'trigrams':
        trigrams = Phrases(bigrams_phraser[all_tokens], min_count=min_count,delimiter=delimiter,scoring=scoring,threshold=threshold)
        trigrams_phraser = Phraser(trigrams)
        
    ## Find bigrams and/or trigrams in each document and replace them with a concatenated token
    for file in lemma_list:
        ngrams_list[file]=[]
        if n=='bigrams':
            for doc in lemma_list[file]:
                new_doc = bigrams_phraser[doc]
                ngrams_list[file].append(new_doc)
        if n=='trigrams':
            for doc in lemma_list[file]:
                new_doc = trigrams_phraser[bigrams_phraser[doc]]
                ngrams_list[file].append(new_doc)

    
    with open((os.path.join(token_dictionary_dir,'ngrams')+'.pickle'), 'wb') as handle:
        pickle.dump(ngrams_list, handle, protocol = pickle.HIGHEST_PROTOCOL)

    ## Business Logic --
    
    ## Return for success ++
    status['status']     = True
    status['msg']        = 'Success'
    status['ngrams_list'] = ngrams_list
    ## Return for success --

    return status 