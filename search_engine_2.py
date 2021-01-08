from tqdm import tqdm
from reader import ReadFile
from nltk import pos_tag
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher
import utils
import pandas as pd
from wordnet import Wordnet


# Wordnet
class SearchEngine:
    def __init__(self, config=None):
        self._config = config
        self._parser = Parse(False)
        self.reader = ReadFile(corpus_path=config.get__corpusPath())
        self._indexer = Indexer(config)
        self.model = None

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
        utils.save_pickle_tuple(tuple_to_save, 'idx_engine2', self._config.get_out_path())

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
        self._indexer.inverted_idx, self._indexer.document_dict = self.load_index('idx_engine2.pkl')
        searcher = Searcher(self._parser, self._indexer, model=self.model)
        # TODO check about K
        query_as_list = self._parser.parse_sentence(query)
        list_copy = list(query_as_list[0])
        tagged_words = pos_tag(list_copy)
        for word in tagged_words:
            wn_tag = Wordnet.get_wordnet_pos(word[1])
            synonym = Wordnet.get_closest_term(word[0], wn_tag)
            if synonym is not None:
                list_copy.append(synonym)
        l_res = searcher.search(list_copy)
        t_ids = [tup[1] for tup in l_res]
        return len(l_res), t_ids
