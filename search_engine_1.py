from tqdm import tqdm

from reader import ReadFile
from configuration import ConfigClass
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher
import utils
import numpy as np
import pandas as pd


# By GloVe Method
class SearchEngineGlove:
    GLOVE_PATH_SERVER = '../../../../glove.twitter.27B.25d.txt'
    GLOVE_PATH_LOCAL = 'glove.twitter.27B.25d.txt'
    # glove_dict = {}
    model = None
    def __init__(self, config=None):
        self._config = config
        self._parser = Parse(False)
        self.reader = ReadFile(corpus_path=config.get__corpusPath())
        self._indexer = Indexer(config)
        self.model = self.initialize_glove_dict()
        self._indexer.set_glove_dict(self.model)

    def initialize_glove_dict(self):
        glove_dict = {}
        with open(self.GLOVE_PATH_LOCAL, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector
        return glove_dict

# with open(GLOVE_PATH_LOCAL, 'r', encoding='utf-8') as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:], "float32")
#         glove_dict[word] = vector


# load_glove_dict()

    def run_engine(self, config):
        """

        :return:
        """

        number_of_documents = 0
        sum_of_doc_lengths = 0

        # r = ReadFile(corpus_path=config.get__corpusPath())
        # p = Parse(config.toStem)
        # indexer = Indexer(config)
        documents_list = self.reader.read_file(file_name=config.get__corpusPath())
        # parquet_documents_list = r.read_folder(config.get__corpusPath())
        # for parquet_file in parquet_documents_list:
        #     documents_list = r.read_file(file_name=parquet_file)
            # Iterate over every document in the file
        for idx, document in tqdm(enumerate(documents_list)):
            # parse the document
            parsed_document = self._parser.parse_doc(document)
            if parsed_document is None:
                continue
            number_of_documents += 1
            sum_of_doc_lengths += parsed_document.doc_length
            # index the document data
            self._indexer.add_new_doc(parsed_document)

        # saves last posting file after indexer has done adding documents.
        self._indexer.save_postings()
        if len(self._indexer.doc_posting_dict) > 0:
            self._indexer.save_doc_posting()
        utils.save_dict(self._indexer.document_dict, "documents_dict", config.get_out_path())
        if len(self._indexer.document_posting_covid) > 0:
            self._indexer.save_doc_covid()

        self._indexer.delete_dict_after_saving()

        # merges posting files.
        self._indexer.merge_chunks()
        utils.save_dict(self._indexer.inverted_idx, "inverted_idx", config.get_out_path())

        dits = {'number_of_documents': number_of_documents, "avg_length_per_doc": sum_of_doc_lengths / number_of_documents}

        utils.save_dict(dits, 'details', config.get_out_path())


    def load_index(self, out_path=''):
        inverted_index = self._indexer.load_index('inverted_index.pkl')
        documents_dict = utils.load_dict("documents_dict", out_path)
        dits = utils.load_dict('details', out_path)
        num_of_docs, avg_length_per_doc = dits['number_of_documents'], dits['avg_length_per_doc']
        return inverted_index, documents_dict, num_of_docs, avg_length_per_doc


    # def search_and_rank_query(self, query, k):
    #     # p = Parse(config.toStem)
    #     query_as_list = self._parser.parse_sentence(query)
    #     searcher = Searcher(self._parser, self._indexer, self.model)
    #     # s = time.time()
    #     relevant_docs, query_glove_vec, query_vec = searcher.relevant_docs_from_posting(query_as_list[0])
    #     # print("Time for searcher: {}".format(time.time() - s))
    #     # s=time.time()
    #     ranked_docs = searcher.ranker.rank_relevant_doc(relevant_docs, query_glove_vec, query_vec)
    #     # print("Time for ranker: {}".format(time.time() - s))
    #     check = searcher.ranker.retrieve_top_k(ranked_docs, k)
    #     return check

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.

    def search(self, query):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results.
        Input:
            query - string.
        Output:
            A tuple containing the number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """
        searcher = Searcher(self._parser, self._indexer, model=self.model)
        # TODO check about K
        return searcher.search(query, 10)


    def main(self, corpus_path=None, output_path='', stemming=False, queries=None, num_docs_to_retrieve=1):
        if queries is not None:
            # config = ConfigClass(corpus_path, output_path, stemming)
            self.run_engine(self._config)

        query_list = handle_queries(queries)
        # inverted_index, document_dict, num_of_docs, avg_length_per_doc = 0self.load_index(output_path)
        # tweet_url = 'http://twitter.com/anyuser/status/'
        # num_of_docs = 10000000
        # avg_length_per_doc = 21.5
        for idx, query in enumerate(query_list):
            docs_list = self.search(query)
            for doc_tuple in docs_list:
                print('tweet id: {}, score: {}'.format(str(doc_tuple[1]), doc_tuple[0]))


def write_to_csv(tuple_list):
    df = pd.DataFrame(tuple_list, columns=['query', 'tweet_id', 'score'])
    df.to_csv('results.csv')


def handle_queries(queries):
    if type(queries) is list:
        return queries

    q = []
    with open(queries, 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n':
                # start = line.find('.')
                # q.append(line[start+2:])
                q.append(line)

    return q