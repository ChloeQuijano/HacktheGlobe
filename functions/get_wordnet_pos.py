from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    """
    This function maps a part of speech tag produced by NLTK to those recognized by WordNet

    @param: pos_tag: part of speech tag as output by nltk.pos_tag
    @return value: status: True for success or False for Failure
                   wordnet_tag: In case the execution is successful, a wordnet compatible pos tag
                   msg: In case of error, a message summarise the error
    """
    ## Status is failure as a default ++
    status   = {
        'status'      : False
        , 'msg'       : 'Default error message'
        , 'wordnet_tag': ''
    }
    ## Status is failure as a default --

    ## Business logic ++

    if pos_tag.startswith('J'):
        wordnet_tag = wordnet.ADJ
    elif pos_tag.startswith('V'):
        wordnet_tag = wordnet.VERB
    elif pos_tag.startswith('N'):
        wordnet_tag = wordnet.NOUN
    elif pos_tag.startswith('R'):
        wordnet_tag = wordnet.ADV
    else:
        wordnet_tag = ''
    ## Business logic --

    ## Return for success ++
    status['status']     = True
    status['msg']        = 'Wordnet tag returned'
    status['wordnet_tag'] = wordnet_tag
    ## Return for success --

    return status