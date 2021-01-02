import search_engine_best
from search_engine_1 import SearchEngineGlove
from configuration import ConfigClass

if __name__ == '__main__':
    search_engine_1 = SearchEngineGlove(config=ConfigClass(output_path='', corpus_path=''))
    search_engine_1.main(queries=["Hello", "World"])
