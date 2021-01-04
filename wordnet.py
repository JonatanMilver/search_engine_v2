from nltk.corpus import wordnet as wn

class Wordnet:
    def __init__(self):
        pass

    def get_closest_term(self, term):
        synset = wn.synsets(term)
        try:
            synonym = synset[0].lemmas()[0].name()
            if term.lower() != synonym.lower():
                return synonym
            return None
        except:
            return None


