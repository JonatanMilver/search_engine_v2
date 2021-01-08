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
class SearchEngine:
    GLOVE_PATH_SERVER = '../../../../glove.twitter.27B.25d.txt'
    GLOVE_PATH_LOCAL = '.\model/model.txt'
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
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector
        return glove_dict

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def build_index_from_parquet(self, fn):
        """
        Reads parquet file and passes it to the parser, then indexer.
        Input:
            fn - path to parquet file
        Output:
            No output, just modifies the internal _indexer object.
        """
        df = pd.read_parquet(fn, engine="pyarrow")
        documents_list = df.values.tolist()
        # Iterate over every document in the file
        number_of_documents = 0
        for idx, document in tqdm(enumerate(documents_list)):
            # parse the document
            parsed_document = self._parser.parse_doc(document)
            if parsed_document is None:
                continue
            number_of_documents += 1
            # index the document data
            self._indexer.add_new_doc(parsed_document)

        tuple_to_save = self._indexer.fix_inverted_index()
        utils.save_pickle_tuple(tuple_to_save, 'idx_engine1', self._config.get_out_path())

        print('Finished parsing and indexing.')

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_precomputed_model(self, model_path):
        """
        Loads a pre-computed model (or models) so we can answer queries.
        This is where you would load models like word2vec, LSI, LDA, etc. and
        assign to self._model, which is passed on to the searcher at query time.
        """
        pass

    def load_index(self, fn):
        return self._indexer.load_index(fn)

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
        self._indexer.inverted_idx, self._indexer.document_dict = self.load_index('idx_engine1.pkl')
        searcher = Searcher(self._parser, self._indexer, model=self.model)
        # TODO check about K
        query_as_list = self._parser.parse_sentence(query)
        l_res = searcher.search(query_as_list[0])
        t_ids = [tup[1] for tup in l_res]
        return len(l_res), t_ids
