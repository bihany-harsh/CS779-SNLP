import spacy
eng_nlp = spacy.load("en_core_web_sm")
from indicnlp.tokenize import indic_tokenize

class Language:
  def __init__(self, lang="en"):
    self.language = lang
    self.word2idx = {}
    self.idx2word = {0: "<SOS>", 1:"<EOS>"}
    self.n_words = 2
    self.word_count = {}

  def add_word(self, word):
    if word not in self.word2idx:
      self.word2idx[word] = self.n_words
      self.idx2word[self.n_words] = word
      self.word_count[word] = 1
      self.n_words += 1
    else:
      self.word_count[word] += 1

  def add_sentence(self, sent):
    tokens = []
    if self.language != "en":
      tokens = [token.text for token in eng_nlp(sent)]
    else:
      tokens = [token for token in indic_tokenize.trivial_tokenize(sent)]
      # tokens = [token for token in sent.split(" ")]

    for token in tokens:
      self.add_word(token)

  def idx_from_sentence(self, sentence):
      return [self.word2idx[word] for word in sentence.split(' ') if word in self.word2idx]

  def sentence_from_idx(self, indices):
      return ' '.join([self.idx2word[idx] for idx in indices])