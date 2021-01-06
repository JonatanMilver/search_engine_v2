from nltk.corpus import lin_thesaurus as lt

class ThesaurusModel:

    def __init__(self):
        pass
    @staticmethod
    def get_synonym(word):
        synonyms_types = lt.synonyms(word[0])
        pos_tag = word[1]
        if pos_tag.startswith('J'):
            synonyms_list = list(synonyms_types[0][1])
        elif pos_tag.startswith('V'):
            synonyms_list = list(synonyms_types[2][1])
        else:
            synonyms_list = list(synonyms_types[1][1])
        if len(synonyms_list) > 0:
            return synonyms_list[0]
        return None
