import math
from collections import OrderedDict, Counter

import numpy as np

import document
import utils

# DO NOT MODIFY CLASS NAME
class Indexer:
    GLOVE_PATH_SERVER = '../../../../glove.twitter.27B.25d.txt'
    GLOVE_PATH_LOCAL = 'glove.twitter.27B.25d.txt'
    MAX_ACCUMULATIVE = 4000000
    POSTING_DICT_SIZE = 200000
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def __init__(self, config):
        # given a term, returns the number of doc/tweets in which he is in
        # term -> df, posting_idx
        self.inverted_idx = {}
        self.document_dict = {}
        self.document_posting = {}
        self.document_posting_covid = {}
        self.doc_posting_covid_counter = 1
        self.document_posting_counter = 1
        self.locations_at_postings = OrderedDict()
        # posting_list example -> [('banana', [doc1, doc2,..]) ...]
        # doc1 -> ('tweet_id', and more details..)
        self.posting_list = []
        self.accumulative_size = 0
        self.num_of_docs = 0
        self.posting_dict = {}
        self.counter_of_postings = 0
        self.global_capitals = {}
        self.entities_dict = Counter()
        self.config = config
        self.doc_posting_counter = 0
        self.doc_posting_dict = {}
        self.merged_dicts = []
        self.glove_dict = {}


    def initialize_glove_dict(self):
        glove_dict = {}
        with open(self.GLOVE_PATH_LOCAL, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector
        return glove_dict

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
        for key_term in document_capitals:
            if key_term not in self.global_capitals:
                self.global_capitals[key_term] = document_capitals[key_term]
            else:
                if not document_capitals[key_term]:
                    self.global_capitals[key_term] = False

        document_entities = document.named_entities

        for entity in document_entities:
            self.entities_dict[entity] += 1
        document_vec = np.zeros(shape=25)
        is_covid = False
        for term in document_dictionary:
            if term == 'covid':
                is_covid = True
            if term in self.glove_dict:
                document_vec += self.glove_dict[term]
        document_vec /= len(document_dictionary)
        # document_vec, # numpy array of size 25 which
        # represents the document in 25 dimensional space(GloVe)
        if is_covid:
            self.document_posting_covid[document.tweet_id] = (document_vec, document_date)
            self.document_dict[document.tweet_id] = ["doc_posting_covid" + str(self.doc_posting_covid_counter), 0]
        else:
            self.doc_posting_dict[document.tweet_id] = (document_vec, document_date)
            self.document_dict[document.tweet_id] = ["doc_posting"+str(self.doc_posting_counter), 0]

        if len(self.doc_posting_dict) == 100000:
            self.save_doc_posting()
        if len(self.document_posting_covid) == 100000:
            self.save_doc_covid()


        # Go over each term in the doc
        for term in document_dictionary.keys():
            try:
                if term in self.inverted_idx:
                    if term not in self.posting_dict:
                        self.posting_dict[term] = None
                    self.inverted_idx[term][0] += 1

                else:
                    self.inverted_idx[term] = [1, str(self.counter_of_postings)]
                    self.posting_dict[term] = None

                insert_tuple = (document.tweet_id,  # tweet id
                                document.doc_length,  # total number of words in tweet
                                document.max_tf,  # number of occurrences of most common term in tweet
                                document.unique_terms,  # number of unique words in tweet
                                document_dictionary[term]/document.doc_length,  #normalized number of times term is in tweet - normalized tf!
                                )
                # if there're no documents for the current term, insert the first document
                if self.posting_dict[term] is None:
                    self.posting_list.append((term, [insert_tuple]))
                    self.posting_dict[term] = len(self.posting_list) - 1
                else:
                    tuple_idx = self.posting_dict[term]
                    self.posting_list[tuple_idx][1].append(insert_tuple)

                # check if posting_dict is full
                if len(self.posting_list) == self.POSTING_DICT_SIZE:
                    self.save_postings()

            except:
                print('problem with the following key {}'.format(term))


    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        return utils.load_dict(fn, self.config.get_out_path())

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        raise NotImplementedError

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

    def save_postings(self):
        o = sorted(self.posting_list, key=lambda x: x[0])
        self.locations_at_postings[str(self.counter_of_postings)] = utils.save_list(o, str(self.counter_of_postings)
                                                                                    , self.config.get_out_path())
        self.counter_of_postings += 1
        self.posting_dict = {}
        self.posting_list = []

    def merge_chunks(self):
        """
        performs a K-way merge on the posting files -> N disk accesses
        writes new posting files to the disk.
        :return:
        """
        saved_chunks = []
        chunks_indices = np.zeros(shape=(len(self.locations_at_postings)), dtype=np.int32)
        chunk_length = self.POSTING_DICT_SIZE // len(self.locations_at_postings) + 1
        #   inserts the chunks into a chunked list
        for key in self.locations_at_postings:
            loaded, offset = utils.load_list(key, self.config.get_out_path(), self.locations_at_postings[key],
                                             chunk_length)
            saved_chunks.append(loaded)
            self.locations_at_postings[key] = offset

        building_list = []
        all_empty = True

        # loops through as long as all postings files didn't finish running.
        while all_empty:
            should_enter = -1

            # loops through as long as one of the chunks is not done.
            while should_enter == -1:
                term_to_enter = self.find_term(saved_chunks, chunks_indices)
                tuples_to_merge = []
                indexes_of_the_indexes_to_increase = []

                # find all tuples that should be merged and the indices should be increased
                for idx, term_idx_in_chunk in enumerate(chunks_indices):
                    if term_idx_in_chunk < len(saved_chunks[idx]) and \
                            saved_chunks[idx][term_idx_in_chunk][0] == term_to_enter:
                        tuples_to_merge.append(saved_chunks[idx][term_idx_in_chunk])
                        indexes_of_the_indexes_to_increase.append(idx)

                merged_list = self.merge_terms_into_one(tuples_to_merge)
                appended_term = merged_list[0]
                should_append = True
                # if it is a named entity and it exists in less than 2 tweets, erase this term.
                if appended_term in self.entities_dict and self.entities_dict[appended_term] < 2:
                    should_append = False
                    self.inverted_idx.pop(appended_term, None)
                # update terms with capital letters
                if appended_term in self.global_capitals and self.global_capitals[appended_term]:
                    merged_list = [appended_term.upper(), merged_list[1]]
                    inverted_val = self.inverted_idx[appended_term]
                    self.inverted_idx.pop(appended_term, None)
                    self.inverted_idx[appended_term.upper()] = inverted_val
                appended_term = merged_list[0]
                if appended_term in self.inverted_idx and self.inverted_idx[appended_term][0] == 1:
                    should_append = False
                    self.inverted_idx.pop(appended_term, None)
                if should_append:
                    self.accumulative_size += len(merged_list[1])
                    idf = self.calculate_idf(merged_list[0])
                    merged_dict = {}
                    for idx,tweet_tuple in enumerate(merged_list[1]):
                        tf_idf = tweet_tuple[4]*idf
                        new_tweet_tuple = (tweet_tuple[1],
                                           tweet_tuple[2],
                                           tweet_tuple[3],
                                           tf_idf)
                        merged_list[1][idx] = new_tweet_tuple
                        merged_dict[tweet_tuple[0]] = new_tweet_tuple
                        self.document_dict[tweet_tuple[0]][1] += math.pow(tf_idf, 2)
                    merged_list[1] = merged_dict
                    building_list.append(merged_list)
                    self.inverted_idx[merged_list[0]][1] = str(self.counter_of_postings)

                # increase the indices that the tuple at the specific location have been inserted to the new posting
                for idx in indexes_of_the_indexes_to_increase:
                    chunks_indices[idx] += 1

                should_enter = self.update_should_enter(saved_chunks, chunks_indices)

                # saving happens as soon as the size reaches given max size of the final posting
                if self.accumulative_size >= self.MAX_ACCUMULATIVE:
                    self.merged_dicts.append(str(self.counter_of_postings))
                    utils.save_list(building_list, str(self.counter_of_postings), self.config.get_out_path())
                    self.accumulative_size = 0
                    self.counter_of_postings += 1
                    building_list = []
            # loads new chunks into the save_chunks list in the relevant indices.
            for index in should_enter:
                loaded, offset = utils.load_list(str(index), self.config.get_out_path(),
                                                 self.locations_at_postings[str(index)], chunk_length)
                saved_chunks[index] = loaded
                chunks_indices[index] = 0
                self.locations_at_postings[str(index)] = offset

            # checks whether all postings are done.
            all_empty = False
            for chunk in saved_chunks:
                if len(chunk) > 0:
                    all_empty = True
                    break

        # save of the last posting file.
        if len(building_list) > 0:
            self.merged_dicts.append(str(self.counter_of_postings))
            utils.save_list(building_list, str(self.counter_of_postings), self.config.get_out_path())

    def merge_terms_into_one(self, tuples_to_merge):
        """
        merges all tuples with the same term into one tuple in order to be appented to the new list.
        :param tuples_to_merge: list of tuples to be merged.
        :return:
        """
        if len(tuples_to_merge) == 1:
            return list(tuples_to_merge[0])
        ret_tuple = [tuples_to_merge[0][0], []]
        for tup in tuples_to_merge:
            ret_tuple[1].extend(tup[1])
        ret_tuple[1].sort(key=lambda x: x[0])
        return ret_tuple

    def find_term(self, saved_chunks, chunks_indices):
        """
        Find the smallest term, the one that should be appended next.
        :param saved_chunks: list of chunks.
        :param chunks_indices: list of current indices of each chunk.
        :return: the smallest term
        """
        term_to_enter = None
        for idx, chunk in enumerate(saved_chunks):
            if len(chunk) > 0:
                term_to_enter = saved_chunks[idx][chunks_indices[idx]][0]
        for idx in range(len(saved_chunks)):
            if chunks_indices[idx] < len(saved_chunks[idx]) and \
                    saved_chunks[idx][chunks_indices[idx]][0] < term_to_enter:
                term_to_enter = saved_chunks[idx][chunks_indices[idx]][0]
        return term_to_enter

    def update_should_enter(self, saved_chunks, chunks_indices):
        """
        returns -1 if the loop should continue
        otherwise, returns the list of indices of the indices to increase.
        :param saved_chunks: list of chunks.
        :param chunks_indices: list of current indices of each chunk.
        :return:
        """
        load_list = []
        for i in range(len(saved_chunks)):
            if chunks_indices[i] >= len(saved_chunks[i]):
                load_list.append(i)
        if len(load_list) == 0:
            return -1
        return load_list

    def calculate_idf(self, term):
        """
        calculates idf of term
        :param term_data: term information
        :return:
        """
        # to calc idf
        n = self.num_of_docs
        df = self.inverted_idx[term][0]
        idf = math.log10(n / df)
        return idf

    def delete_dict_after_saving(self):
        del self.document_dict
        del self.doc_posting_dict

    def save_doc_posting(self):
        utils.save_dict(self.doc_posting_dict, "doc_posting" + str(self.doc_posting_counter),
                        self.config.get_out_path())
        self.doc_posting_counter += 1
        self.doc_posting_dict = {}

    def save_doc_covid(self):
        utils.save_dict(self.document_posting_covid, "doc_posting_covid" + str(self.doc_posting_covid_counter),
                        self.config.get_out_path())
        self.doc_posting_covid_counter += 1
        self.document_posting_covid = {}

    def set_glove_dict(self, model):
        self.glove_dict = model

    def set_num_of_doc(self, number_of_documents):
        self.num_of_docs = number_of_documents
