

class InvertedIndex:
    """
    Holds the inverted index dictionary.
    All operations on inverted index would happen through this class.
    """
    def __init__(self):
        # {term: [df, {tweet_id: [doc_length, max_tf, number_of_unique_terms, tf_idf, tweet_date]]}}
        self.main_dict = {}

    def __getitem__(self, term):
        return self.main_dict[term]

    def __iter__(self):
        return iter(self.main_dict)

    def keys(self):
        return self.main_dict.keys()

    def items(self):
        return self.main_dict.items()

    def values(self):
        return self.main_dict.values()

    def insert(self, term, tweet_id, doc_length, max_tf, number_of_unique_terms, tf_idf, tweet_date):
        """
        inserts new doc with tweet_id to key-term
        :param term: str - the dict key
        :param tweet_id: int - doc's id
        :param doc_length: int - num of words in the doc
        :param max_tf: int - max frequency of a word in doc.
        :param number_of_unique_terms: int - num of unique tems
        :param tf_idf: float - at the beginning, holds normalized tf. at the end of building the index, it would hold the tf-idf
        :param tweet_date: datetime - date
        :return:
        """
        if term not in self.main_dict:
            self.main_dict[term] = []
            self.main_dict[term].append(1)
            self.main_dict[term].append({})
        else:
            self.main_dict[term][0] += 1
        self.main_dict[term][1][tweet_id] = [doc_length, max_tf, number_of_unique_terms, tf_idf, tweet_date]

    def insert_entry(self, term, whole_list):
        """
        inserts list of tweets into the dict with key - term
        :param term: str - dict's key
        :param whole_list: list to insert
        :return:
        """
        self.main_dict[term] = whole_list

    def get_tweets_with_term(self, term):
        if term in self.main_dict:
            return self.main_dict[term][1]

    def remove(self, term):
        if term in self.main_dict:
            del self.main_dict[term]

    def get_df(self, term):
        if term in self.main_dict:
            return self.main_dict[term][0]

    def get_doc_length(self, term, tweet_id):
        if term in self.main_dict:
            return self.main_dict[term][1][tweet_id][0]

    def get_term_info(self, term):
        if term in self.main_dict:
            return self.main_dict[term]

    def get_max_tf(self, term, tweet_id):
        if term in self.main_dict:
            return self.main_dict[term][1][tweet_id][1]

    def get_number_of_unique_terms(self, term, tweet_id):
        if term in self.main_dict:
            return self.main_dict[term][1][tweet_id][2]

    def get_tf_idf(self, term, tweet_id):
        if term in self.main_dict:
            return self.main_dict[term][1][tweet_id][3]

    def get_tweet_date(self, term, tweet_id):
        if term in self.main_dict:
            return self.main_dict[term][1][tweet_id][4]

    def set_tf_idf(self, term, tweet_id, tf_idf):
        if term in self.main_dict:
            self.main_dict[term][1][tweet_id][3] = tf_idf

