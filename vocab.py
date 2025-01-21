
class Vocab:
    sos_token = 0
    eos_token = 1
    pad_token = 2
    unk_token = 3

    def __init__(self,name):
        self.name = name
        self.word2index = {"SOS":0, "EOS":1,"PAD":2,"UNK":3}
        self.word2count = {"SOS":0, "EOS":0,"PAD":0,"UNK":0}
        self.index2word = {0:"SOS",1:"EOS",2:"PAD",3:"UNK"}
        self.n_words = len(self.index2word)

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):

        if word not in self.word2index:
          self.word2index[word] = self.n_words
          self.word2count[word] = 1
          self.index2word[self.n_words] = word
          self.n_words += 1
        else:
          self.word2count[word] +=1

    def trim_vocab(self, top_k=50000):
        # 상위 top_k 개수만 남기도록 정렬
        sorted_words = sorted(self.word2count.items(), key=lambda item: item[1], reverse=True)[:top_k]

        # 기존 속성들 초기화
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

        # 필수 토큰 유지
        for token, index in [("SOS", 0), ("EOS", 1), ("PAD", 2), ("UNK", 3)]:
            self.word2index[token] = index
            self.word2count[token] = 0
            self.index2word[index] = token
            self.n_words += 1

        # 상위 top_k 단어만 새롭게 추가
        for word, count in sorted_words:
            if word not in self.word2index:  # 필수 토큰이 아닌 경우
                self.word2index[word] = self.n_words
                self.word2count[word] = count
                self.index2word[self.n_words] = word
                self.n_words += 1
    
    def to_dict(self):
        return {
            "name": self.name,
            "word2index": self.word2index,
            "word2count": self.word2count,
            "index2word": self.index2word,
            "n_words": self.n_words
        }
