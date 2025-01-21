import torch
from torch.utils.data import Dataset, DataLoader
from vocab import Vocab
import re
import pickle
import json

class NMT_Dataset(Dataset):
    def __init__(self, input_tensors, output_tensors):
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.output_tensors[idx]


def normalize_string(s):
    # s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_languages(data_type, use_saved_vocab): ## train / dev te

    chunk_size=1024*1024 # 1MB 단위로 처리
    # path = "/hdd/user15/RNN/dataset/"
    # if data_type == "train":
    #     en_path = path + f"processed_train/train_en.txt"
    #     de_path = path + f"processed_train/train_de.txt"
    # elif data_type == "dev":
    #     en_path = path + f"processed_dev/newstest2013.en"
    #     de_path = path + f"processed_dev/newstest2013.de"
    # elif data_type == "test":
    #     en_path = path + f"processed_test/newstest2014-deen-src.en.2"
    #     de_path = path + f"processed_test/newstest2014-deen-ref.de.2"

    path = "/home/user15/RNN/dataset5/"
    if data_type == "train":
        en_path = path + f"train.en"
        de_path = path + f"train.de"
    elif data_type == "dev":
        en_path = path + f"newstest_2013.en"
        de_path = path + f"newstest_2013.de"
    elif data_type == "test":
        en_path = path + f"newstest2014.en"
        de_path = path + f"newstest2014.de"

    # Read and parse the text file
    with open(en_path, 'r', encoding = 'utf-8') as f1, open(de_path,'r', encoding = 'utf-8') as f2:
        #pairs = [[normalize_string(en_line.strip()), normalize_string(de_line.strip())] for en_line, de_line in zip(f1, f2)]
        pairs = [[en_line.strip(), de_line.strip()] for en_line, de_line in zip(f1, f2)]

        if use_saved_vocab == True:
            with open('vocab_global/global_input_vocab.pkl', 'rb') as f1, open('vocab_global/global_output_vocab.pkl', 'rb') as f2:
                input_vocab = pickle.load(f1)
                output_vocab = pickle.load(f2)
        else:
            input_vocab = Vocab('en')
            output_vocab = Vocab('de')
            
    return input_vocab, output_vocab, pairs
    
    
    
def filter_pair(pair,max_len):
    is_good_pair = (len(pair[0].split(' ')) <= max_len) and (len(pair[1].split(' ')) <= max_len)
    return is_good_pair

def filter_pairs(pairs,max_len):
    return [pair for pair in pairs if filter_pair(pair,max_len)]

def prepare_data(data_type, max_len, use_saved_vocab): ## train / dev / test
    input_vocab, output_vocab, pairs = read_languages(data_type, use_saved_vocab)
    pairs = filter_pairs(pairs,max_len)


    if use_saved_vocab == True:
        pass
    else:
        for pair in pairs:
            input_vocab.index_words(pair[0])
            output_vocab.index_words(pair[1])
            
        if input_vocab.n_words > 50000:
            input_vocab.trim_vocab(50000)
        if output_vocab.n_words > 50000:
            output_vocab.trim_vocab(50000)
        # 저장
        with open('vocab_global/global_input_vocab.pkl', 'wb') as f1, open('vocab_global/global_output_vocab.pkl', 'wb') as f2:
            pickle.dump(input_vocab, f1)  # to_dict 메서드를 사용하여 dict로 변환
            pickle.dump(output_vocab, f2)  # to_dict 메서드를 사용하여 dict로 변환
            
    return input_vocab, output_vocab, pairs


# read_languages("train") ## 약 600MB



def indexes_from_sentence(vocab, sentence, data_type):
    ## indexing 단계
    ## 현재 입력 단어들을 Vocab에 입력해놓은 상태임.
    out_sentence = []
    for word in sentence.split(' '):
        if word not in vocab.word2index:
            out_sentence.append(vocab.word2index["UNK"])
        else:
            out_sentence.append(vocab.word2index[word])
    return out_sentence

def tensor_from_sentence(vocab, sentence, is_output, max_len, data_type):
    _indexes = indexes_from_sentence(vocab, sentence, data_type)

    if is_output:
        _indexes.append(vocab.eos_token)
        if len(_indexes) < max_len + 1:
            _indexes += [vocab.pad_token] * (max_len + 1 - len(_indexes))
        else:
            _indexes = _indexes[:max_len+1]

    else:
        if len(_indexes) < max_len:
            _indexes += [vocab.pad_token] * (max_len - len(_indexes))
        else:
            _indexes = _indexes[:max_len]

    return torch.tensor(_indexes, dtype = torch.long)


def tensor_from_pair(pairs, input_vocab, output_vocab, max_len, data_type):

    batch_size = len(pairs)

    # 미리 텐서를 할당 (배치 크기, 최대 문장 길이)
    input_tensors = torch.zeros((batch_size, max_len), dtype=torch.long)
    output_tensors = torch.zeros((batch_size, max_len+1), dtype=torch.long)

    for i, pair in enumerate(pairs):
        input_indexes = tensor_from_sentence(input_vocab, pair[0],False, max_len, data_type)
        output_indexes = tensor_from_sentence(output_vocab, pair[1],True, max_len, data_type)

        # 미리 할당한 텐서에 각 문장 인덱스를 저장
        input_tensors[i] = input_indexes
        output_tensors[i] = output_indexes

    return input_tensors, output_tensors