import math
from collections import Counter
from inverted_index import InvertedIndex
import numpy as np
import document
import utils

# DO NOT MODIFY CLASS NAME
class Indexer:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def __init__(self, config):
        # An object representing the inverted_index: {term: [df, {tweet_id: list of tweet information...}...]..}
        self.inverted_idx = InvertedIndex()
        # Represents the GloVe vector
        self.document_dict = {}
        self.num_of_docs = 0
        self.global_capitals = {}
        self.entities_dict = Counter()
        self.config = config
        self.glove_dict = {}


    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def add_new_doc(self, document):
        """
        This function perform indexing process for a document object.
        Saved information is captures via two dictionaries ('inverted index' and 'posting')
        :param document: a document need to be indexed.
        :return: -
        """
        document_dictionary = document.term_doc_dictionary
        document_capitals = document.capital_letter_indexer
        document_date = document.tweet_date

        # Handling CAPITALS
        for key_term in document_capitals:
            if key_term not in self.global_capitals:
                self.global_capitals[key_term] = document_capitals[key_term]
            else:
                if not document_capitals[key_term]:
                    self.global_capitals[key_term] = False
        # Handling NAMED ENTITIES
        document_entities = document.named_entities
        for entity in document_entities:
            self.entities_dict[entity] += 1

        # Initializing Glove vector
        document_vec = np.zeros(shape=25)

        # Go over each term in the doc
        for term in document_dictionary.keys():
            try:
                # Building GloVe vector
                if term in self.glove_dict:
                    document_vec += self.glove_dict[term]

                # At the beginning - tf_idf(document_dictionary[term]/document.doc_length) will hold normalized tf
                # tf_idf will be changed at fixing_index function to the real tf_idf
                self.inverted_idx.insert(term, document.tweet_id, document.doc_length,
                                         document.max_tf, document.unique_terms,
                                         document_dictionary[term]/document.max_tf, document_date)

            except:
                print('problem with the following key {}'.format(term))

        document_vec /= len(document_dictionary)
        # document_vec, # numpy array of size 25 which
        # represents the document in 25 dimensional space(GloVe)
        self.document_dict[document.tweet_id] = [0]*2
        self.document_dict[document.tweet_id][0] = document_vec
        self.num_of_docs += 1

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        return utils.load_pickle_tuple(fn, self.config.get_out_path())

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        obj_tuple = (self.inverted_idx, self.document_dict)
        utils.save_pickle_tuple(obj_tuple, fn, self.config.get_out_path())

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _is_term_exist(self, term):
        """
        Checks if a term exist in the dictionary.
        """
        return term in self.postingDict

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def get_term_posting_list(self, term):
        """
        Return the posting list from the index for a term.
        """
        return self.postingDict[term] if self._is_term_exist(term) else []

    def fix_inverted_index(self):
        for term in list(self.inverted_idx.main_dict):
            should_append = True
            # if it is a named entity and it exists in less than 2 tweets, erase this term.
            if term in self.entities_dict and self.entities_dict[term] < 2:
                should_append = False
                self.inverted_idx.remove(term)
            # update terms with capital letters
            if term in self.global_capitals and self.global_capitals[term]:
                term_info = self.inverted_idx.get_term_info(term)
                self.inverted_idx.remove(term)
                term = term.upper()
                self.inverted_idx.insert_entry(term, term_info)
                # TODO check the amount of min df
            if term in self.inverted_idx.main_dict and self.inverted_idx.get_df(term) < 20:
                should_append = False
                self.inverted_idx.remove(term)
            if should_append:
                term_idf = self.calculate_idf(term)
                tweets = self.inverted_idx.get_tweets_with_term(term)
                # for each tweet, update current term's tf-idf
                for tweet_id in tweets:
                    tf_idf = self.inverted_idx.get_tf_idf(term, tweet_id) * term_idf
                    self.inverted_idx.set_tf_idf(term, tweet_id, tf_idf)
                    self.document_dict[tweet_id][1] += math.pow(tf_idf, 2)
        return self.inverted_idx, self.document_dict

    def calculate_idf(self, term):
        """
        calculates idf of term
        :param term: term for calculation
        :return:
        """
        # to calc idf
        n = self.num_of_docs
        df = self.inverted_idx[term][0]
        idf = math.log10(n / df)
        return idf

    def set_glove_dict(self, model):
        self.glove_dict = model

    def set_num_of_doc(self, number_of_documents):
        self.num_of_docs = number_of_documents
