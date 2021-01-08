from nltk.corpus import wordnet as wn


class Wordnet:
    def __init__(self):
        pass

    @staticmethod
    def get_closest_term(term, wn_tag):
        try:
            synset = wn.synsets(term, pos=wn_tag)[0]
            synonym = synset.lemmas()[0].name()
            if term.lower() != synonym.lower():
                return synonym
            return None
        except:
            return None

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """
        gets part-of-speech tag and translates it to wordnet tag
        :param treebank_tag: tag received from pos tagging
        :return: wordnet tag
        """
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN


