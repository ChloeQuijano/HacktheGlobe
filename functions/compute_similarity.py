import pandas as pd
import gensim 
import seaborn as sns

def compute_similarity(params):
    """
    This function .
    @params: document_dictionary: dictionary of text documents (title = key and text = value) to be analyzed
    @params: corpus: index of one document of interest
    @params: query_doc: Directory where outputs are saved.
    @return value: status: True for success or False for Failure
                   tf_idf: Most relevant ngrams in a document selected for analysis.
                   similarity_index: In case the execution is successful, a similarity matrix for all documents is created.
                   query_similarity: In case of error, a similarity matrix for the query document relative to all other documents/
                   is created. 
    """
    
    ## Status is failure as a default ++
    status = {
        'status'      : False
        ,'msg'       : 'Default error message'
        ,'tf_idf': None
        ,'similarity_index': None
        ,'query_similiarity': None
    }
    ## Status is failure as a default --

    dictionary = params.get('document_dictionary', {})
    query_doc   = params.get('query_doc', None)
    corpus = params.get('corpus', None)
    
    ## Error handling ++
    if {} == dictionary:
        status['status'] = False
        status['msg']    = 'No gensim dictionary provided.'
        return status
    
    if None == corpus:
        status['status'] = False
        status['msg']    = 'No gensim corpus provided.'
        return status

    if None == query_doc:
        query_doc = 0
        print("No query doc provided. Assigned the default of Document 0.")
    
#     if None == output_dir:
#         output_dir = './'
                          
    ## Error handling --
 
    tf_idf = gensim.models.TfidfModel(corpus)
    
    sims = gensim.similarities.MatrixSimilarity(tf_idf[corpus], num_features=len(dictionary))
    
    sims_doc = sims[tf_idf[corpus[query_doc]]]

    similarity = pd.DataFrame(sims_doc, columns = ['Document ' + str(query_doc)])


       
    ## Return for success ++
    status['status']     = True
    status['msg']        = 'Dictionary and corpus for documents created.'
    status['tf_idf'] = tf_idf
    status['similarity_index'] = sims
    status['query_similarity'] = similarity
    ## Return for success --
    
    return status
