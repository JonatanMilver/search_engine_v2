import math
from collections import Counter

from ranker import Ranker
import utils
import numpy as np
from inverted_index import InvertedIndex


# DO NOT MODIFY CLASS NAME
class Searcher:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit. The model 
    # parameter allows you to pass in a precomputed model that is already in 
    # memory for the searcher to use such as LSI, LDA, Word2vec models. 
    # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
    def __init__(self, parser, indexer, model=None):
        self.config = indexer.config
        self._parser = parser
        self._indexer = indexer
        self.number_of_docs = indexer.num_of_docs
        self._model = model
        # self.inverted_index, self.document_dict = self._indexer.load_index("idx_engine1.pkl")
        self.inverted_index, self.document_dict = self._indexer.inverted_idx, self._indexer.document_dict

        self.glove_dict = self._indexer.glove_dict
        use_glove = True
        if len(self.glove_dict) == 0:
            use_glove = False
        self.ranker = Ranker(self.config, use_glove)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query, k=None):
        """ 
        Executes a query over an existing index and returns the number of 
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            A tuple containing the number of relevant search results, and 
            a list of tweet_ids where the first element is the most relavant 
            and the last is the least relevant result.
        """
        relevant_docs, query_glove_vec, square_w_iq = self.relevant_docs_from_posting(query)
        ranked_docs = self.ranker.rank_relevant_doc(relevant_docs, query_glove_vec, square_w_iq)
        top_k = self.ranker.retrieve_top_k(ranked_docs, k)
        return top_k

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def relevant_docs_from_posting(self, query_as_list):
        """
        This function loads the posting list and count the amount of relevant documents per term.
        :param query_as_list: parsed query tokens
        :return: dictionary of relevant documents mapping doc_id to document frequency.
        """

        term_to_indices = {}
        max_tf = 0
        query_glove_vec = np.zeros(shape=25)

        for idx, term in enumerate(query_as_list):
            if term in self.glove_dict:
                query_glove_vec += self.glove_dict[term]

            try:
                if term in self.inverted_index:

                    if term not in term_to_indices:

                        idx_set = {idx}
                        if len(idx_set) > max_tf:
                            max_tf = len(idx_set)
                        term_to_indices[term] = idx_set

                    else:  # term already in term dict, so only update it's index list
                        term_to_indices[term].add(idx)
                        if len(term_to_indices[term]) > max_tf:
                            max_tf = len(term_to_indices[term])

                else:  # term is un-known
                    idx_set = {idx}
                    if len(idx_set) > max_tf:
                        max_tf = len(idx_set)
                    term_to_indices[term] = idx_set

            except:
                print('term {} not found in inverted index'.format(term))

        query_glove_vec /= len(query_as_list)

        p = 0.45
        min_num_of_words_to_relevent = int(len(query_as_list) * p)
        pre_doc_dict = {}
        pre_doc_dict_counter = Counter()

        relevant_docs = {}
        w_iq_square = 0
        for term, term_indices in term_to_indices.items():

            term_tf_idf = ((len(term_indices)/len(query_as_list))*self.calc_idf(term))
            w_iq_square += math.pow(term_tf_idf, 2)

            try:
                # if doc_list is not None:
                if term in self.inverted_index:
                    # for doc_tuple in doc_list.items():
                    for tweet_id in self.inverted_index[term][1]:
                        pre_doc_dict_counter[tweet_id] += 1
                        if tweet_id not in pre_doc_dict:
                            # example - > tf_idf_vec
                            # [[tf1, tf2...]
                            #  [idf1, idf2...]]
                            tf_idf_numarator = 0
                            tf_idf_denomenator = math.sqrt(self.document_dict[tweet_id][1])
                            tweet_doc_length = self.inverted_index.get_doc_length(term, tweet_id)
                            glove_vec = self.document_dict[tweet_id][0]
                            tweet_date = self.inverted_index.get_tweet_date(term, tweet_id)

                            pre_doc_dict[tweet_id] = [tf_idf_numarator, tf_idf_denomenator, tweet_doc_length, glove_vec, tweet_date]

                        pre_doc_dict[tweet_id][0] += self.inverted_index.get_tf_idf(term, tweet_id) * term_tf_idf

                        if tweet_id not in relevant_docs and \
                            pre_doc_dict_counter[tweet_id] >= min_num_of_words_to_relevent:
                            relevant_docs[tweet_id] = pre_doc_dict[tweet_id]

            except:
                print('term {} not found in posting'.format(term))

        return relevant_docs, query_glove_vec, math.sqrt(w_iq_square)

    def calculate_tf(self, tweet_term_tuple):
        """
        calculates term frequency.
        :param tweet_term_tuple: tuple containing all information of the tweet of the term.
        :return:
        """
        # to calc normalize tf
        num_of_terms_in_doc = tweet_term_tuple[1]
        frequency_term_in_doc = tweet_term_tuple[2]
        tf = frequency_term_in_doc / num_of_terms_in_doc

        return tf

    def calculate_idf_BM25(self, term_data):
        """
        calculates idf according to BM25 algorithm.
        :param term_data:
        :return:
        """
        n = self.number_of_docs
        df = term_data[0]
        idf = math.log(((n - df + 0.5) / (df + 0.5)) + 1)
        return idf

    def calc_idf(self, term):
        """
        calculates idf of term
        :param term: term
        :return:
        """
        # to calc idf
        n = self.number_of_docs
        # df = term_data[0]
        if term not in self.inverted_index:
            return 0
        df = self.inverted_index[term][0]
        idf = math.log10(n / df)
        return idf

