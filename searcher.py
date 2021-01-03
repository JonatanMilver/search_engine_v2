import math
from collections import Counter

from ranker import Ranker
import utils
import numpy as np


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
        self.number_of_docs, self.avg_length_per_doc = self.load_details()
        self.document_dict = utils.load_dict("documents_dict", self.config.get_out_path())
        self._ranker = Ranker(self.avg_length_per_doc, self.document_dict, self.config)
        self._model = model
        self.inverted_index = self._indexer.load_index('inverted_idx')

        self.term_to_doclist = {}
        self.glove_dict = model

        self.ranker = Ranker(self.avg_length_per_doc, self.document_dict, self.config)

    def load_details(self):
        dits = utils.load_dict('details', self.config.get_out_path())
        return  dits['number_of_documents'], dits['avg_length_per_doc']

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
        query_as_list = self._parser.parse_sentence(query)
        # searcher = Searcher(self._parser, self._indexer, self.glove_dict)
        # s = time.time()
        relevant_docs, query_glove_vec, square_w_iq = self.relevant_docs_from_posting(query_as_list[0])
        # print("Time for searcher: {}".format(time.time() - s))
        # s=time.time()[
        ranked_docs = self.ranker.rank_relevant_doc(relevant_docs, query_glove_vec, square_w_iq)
        # print("Time for ranker: {}".format(time.time() - s))
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
        # qterm_to_idf = {}
        query_glove_vec = np.zeros(shape=25)
        # query_vec = np.zeros(shape=(2, len(query_as_list)))
        for idx, term in enumerate(query_as_list):
            if term in self.glove_dict:
                query_glove_vec += self.glove_dict[term]
            try:  # an example of checks that you have to do
                if term in self.inverted_index:

                    # qterm_to_idf[term] = self.calculate_idf(self.inverted_index[term])
                    # query_vec[1, idx] = qterm_to_idf[term]
                    if term not in self.term_to_doclist:
                        # all documents having this term is not in the term dict,
                        # so load the appropriate postings and load them
                        curr_posting = utils.load_dict(self.inverted_index[term][1], self.config.get_out_path())

                        doc_list = curr_posting[term]
                        idx_set = {idx}
                        self.term_to_doclist[term] = [idx_set, doc_list]
                        for i in range(idx + 1, len(
                                query_as_list)):  # check if any other terms in query are the same posting to avoid loading it more than once
                            if query_as_list[i] in curr_posting:
                                doc_list = curr_posting[query_as_list[i]]
                                idx_set = {i}
                                self.term_to_doclist[query_as_list[i]] = [idx_set, doc_list]

                    else:  # term already in term dict, so only update it's index list
                        self.term_to_doclist[term][0].add(idx)

                else:  # term is un-known, so
                    # qterm_to_idf[term] = 0
                    doc_list = None
                    idx_set = {idx}
                    self.term_to_doclist[term] = [idx_set, doc_list]

            except:
                print('term {} not found in inverted index'.format(term))

        query_glove_vec /= len(query_as_list)

        p = 0.35
        min_num_of_words_to_relevent = int(len(query_as_list) * p)
        pre_doc_dict = {}
        pre_doc_dict_counter = Counter()

        relevant_docs = {}
        w_iq_square = 0
        for term_to_docs in self.term_to_doclist.items():
            term = term_to_docs[0]

            term_indices = term_to_docs[1][0]
            doc_list = term_to_docs[1][1]

            term_tf_idf = ((len(term_indices)/len(query_as_list))*self.calc_idf(term))

            w_iq_square += math.pow(term_tf_idf, 2)

            try:
                if doc_list is not None:
                    for doc_tuple in doc_list.items():

                        tweet_id = doc_tuple[0]
                        tweet_details = doc_tuple[1]
                        tweet_doc_length = tweet_details[0]
                        pre_doc_dict_counter[tweet_id] += 1

                        if tweet_id not in pre_doc_dict:
                            # example - > tf_idf_vec
                            # [[tf1, tf2...]
                            #  [idf1, idf2...]]
                            tf_idf = 0
                            pre_doc_dict[tweet_id] = [tf_idf, tweet_doc_length]

                        pre_doc_dict[tweet_id][0] += tweet_details[3] * term_tf_idf

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

    # def calculate_idf(self, term_data):
    #     """
    #     calculates idf of term
    #     :param term_data: term information
    #     :return:
    #     """
    #     # to calc idf
    #     n = self.number_of_docs
    #     df = term_data[0]
    #     idf = math.log10(n / df)
    #     return idf

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
        df = self.inverted_index[term][0]
        idf = math.log10(n / df)
        return idf

