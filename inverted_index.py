

class InvertedIndex:
    def __init__(self):
        # {term: [df, {tweet_id: [doc_length, max_tf, number_of_unique_terms, tf_idf, cosine_tweet_denom, tweet_date]]}}
        self.main_dict = {}

    def insert(self, term, tweet_id, doc_length, max_tf, number_of_unique_terms, tf_idf, cosine_tweet_denom, tweet_date):
        if term not in self.main_dict:
            self.main_dict[term] = []
            self.main_dict[term].append(1)
            self.main_dict[term].append({})
        else:
            self.main_dict[term][0] += 1
        self.main_dict[term][1][tweet_id] = [doc_length, max_tf, number_of_unique_terms, tf_idf, cosine_tweet_denom, tweet_date]

    def remove(self, term):
        if term in self.main_dict:
            del self.main_dict[term]

    def get_df(self, term):
        if term in self.main_dict:
            return self.main_dict[term][0]

    def get_doc_length(self, term, tweet_id):
        if term in self.main_dict:
            return self.main_dict[term][1][tweet_id][0]

    def get_max_tf(self, term, tweet_id):
        if term in self.main_dict:
            return self.main_dict[term][1][tweet_id][1]

    def get_number_of_unique_terms(self, term, tweet_id):
        if term in self.main_dict:
            return self.main_dict[term][1][tweet_id][2]

    def get_tf_idf(self, term, tweet_id):
        if term in self.main_dict:
            return self.main_dict[term][1][tweet_id][3]

    def get_cosine_tweet_denominator(self, term, tweet_id):
        if term in self.main_dict:
            return self.main_dict[term][1][tweet_id][4]

    def get_tweet_date(self, term, tweet_id):
        if term in self.main_dict:
            return self.main_dict[term][1][tweet_id][5]

    def set_tf_idf(self, term, tweet_id, tf_idf):
        if term in self.main_dict:
            self.main_dict[term][1][tweet_id][3] = tf_idf
