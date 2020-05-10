import gensim 
import os
import pickle

def create_dictionary_corpus(params):
    """
    This function .
    @params: document_list: dictionary of text documents (title = key and text = value) to be analyzed
    @params: query_doc: index of one document of interest
    @params: output_dir: Directory where outputs are saved.
    @return value: status: True for success or False for Failure
                   dictionary: In case the execution is successful, a gensim dictionary is created.
                   corpus: In case the execution is successful, a gensim corpus is created.
                   msg: In case of error, a message summarise the error status 
    """
    
    ## Status is failure as a default ++
    status = {
        'status'      : False
        ,'msg'       : 'Default error message'
        ,'dictionary': None
        ,'corpus': None
    }
    ## Status is failure as a default --

    compare_docs = params.get('document_dictionary', {})
    query_doc   = params.get('query_doc', None)
    output_dir = params.get('output_dir', None)
    
    ## Error handling ++
    if {} == compare_docs:
        status['status'] = False
        status['msg']    = 'No dictionary of documents to be analysed provided.'
        return status

    if None == query_doc:
        query_doc = 0
        print("No query doc provided. Assigned the default of Document 0.")
    
    if None == output_dir:
        output_dir = './'
                          
    ## Error handling --
 
    flat_text_list = []
    for document in compare_docs.values():
        for items in document:
            flat_text_list.append(items)

    query_doc = query_doc

    dictionary = gensim.corpora.Dictionary([document for document in flat_text_list])
        
    with open((os.path.join(output_dir,'document_dictionary')+'.pickle'), 'wb') as handle:
        pickle.dump(dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)

    corpus = [dictionary.doc2bow(item) for item in flat_text_list]
    
    with open((os.path.join(output_dir,'corpus')+'.pickle'), 'wb') as handle:
        pickle.dump(corpus, handle, protocol = pickle.HIGHEST_PROTOCOL)

    ## Return for success ++
    status['status']     = True
    status['msg']        = 'Dictionary and corpus for documents created.'
    status['dictionary'] = dictionary
    status['corpus'] = corpus
    ## Return for success --
    
    return status
