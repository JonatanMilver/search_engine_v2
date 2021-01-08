import bisect
import math

import numpy as np
from numpy.linalg import norm
import utils



class Ranker:
    def __init__(self, config, useGlove):
        # self.avg_length_per_doc = avg_length
        self.loaded_doc_postings = {}  # key - tweet_id , value - the tweet's vector and the tweet_date
        self.config = config
        self.useGlove = useGlove

    def rank_relevant_doc(self, relevant_doc, query_glove_vec, square_w_iq):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet_id (full_text) and query.
        :param relevant_doc: dictionary of documents that contains at least one term from the query.
        :param query_glove_vec: Vector of 25 representing the query.
        :param square_w_iq: sqrt of sigma((w_iq)^2)
        :return: sorted list of documents by score
        """

        ret = []
        key_list = list(relevant_doc.keys())
        for idx, tweet_id in enumerate(key_list):

            # holds sigma w_ij*w_iq
            sigma_weights_query_doc = relevant_doc[tweet_id][0]
            # holds sigma((w_ij)^2) for current tweet.
            tweet_part_denominator_cosine = relevant_doc[tweet_id][1]

            doc_length = relevant_doc[tweet_id][2]
            glove_vec = relevant_doc[tweet_id][3]
            tweet_date = relevant_doc[tweet_id][4]

            calculated_score = self.calc_score(sigma_weights_query_doc, doc_length, glove_vec, query_glove_vec, square_w_iq, tweet_part_denominator_cosine)
            tweet_tuple = (calculated_score, tweet_id, tweet_date)
            bisect.insort(ret, tweet_tuple)

        return ret

    def retrieve_top_k(self, sorted_relevant_doc, k=1):
        """
        return a list of top K tweets based on their ranking from highest to lowest
        :param sorted_relevant_doc: list of all candidates docs.
        :param k: Number of top document to return
        :return: list of relevant document
        """

        if k is None or k > len(sorted_relevant_doc):
            return sorted(sorted_relevant_doc, key=lambda x: (x[0], x[2]), reverse=True)

        return sorted(sorted_relevant_doc, key=lambda x: (x[0], x[2]))[-k:]

    def calc_score(self, sigma_weights_query_doc, doc_length, glove_vec, query_glove_vec, sqaure_w_iq, tweet_part_denominator_cosine):
        """

        :param sigma_weights_query_doc:
        :param glove_vec:
        :param vec: a 2xlen(query) numpy matrix, first row holds tf data,
                                           secoend row holds idf data
        :param doc_length:
        :return: calculated score of similarity between the represented tweet and the query
        """
        if self.useGlove:
            w_cos_weight = 0.8
            glove_weight = 0.2
        else:
            w_cos_weight = 1
            glove_weight = 0


        word_cosine = w_cos_weight * self.cosine(sigma_weights_query_doc, sqaure_w_iq, tweet_part_denominator_cosine)
        # bm25_score = bm25_weight * self.calc_BM25(bm25_vec, doc_length)
        glove_cosine = glove_weight * self.glove_cosine(glove_vec, query_glove_vec)

        score = word_cosine + glove_cosine

        # if score > 0.85:
        # print("{} : word cosine: {} , glove cosine: {}, total score {}".format(tweet_id,word_cosine,glove_cosine, score))

        return score

    # def calc_BM25(self, vec, doc_length):
    #     # BM25 score calculation
    #     score = 0
    #     k = 1.2
    #     b = 0.75
    #     for column in vec.T:
    #         idf = column[1]
    #         tf = column[0]
    #
    #         score += (idf * tf * (k + 1)) / (tf + k * (1 - b + b * (doc_length / self.avg_length_per_doc)))
    #
    #     return score

    def cosine(self, numerator, query_part_denominator, tweet_part_denominator):
        denominator = query_part_denominator * tweet_part_denominator
        if denominator == 0 or numerator == 0:
            return 0
        return numerator / denominator

    def glove_cosine(self, v1, v2):
        numenator = np.dot(v1, v2)
        denominator = norm(v1) * norm(v2)
        if denominator == 0 or numenator == 0:
            return 0
        return numenator / denominator
