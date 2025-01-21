
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import torch.nn.functional as F
import json
from model_att import Seq2Seq
from dataset2 import NMT_Dataset, prepare_data, tensor_from_pair

from tqdm import tqdm




import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args("")

## Device ##
args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

## Eval ##
args.criterion = torch.nn.CrossEntropyLoss(ignore_index=2)  # Optional ignore padding



## Data sttting parameter ##
args.is_shuffle = True

## Gradient/Optimizer ##
args.max_grad_norm = 5
args.learning_rate_gamma = 0.5

## Path ##
args.best_loss_model_path = 'model_if/best_loss_model.pth'


## Model ##
args.hidden_dimension = 1000
args.max_len = 50
args.n_layers = 4
args.dropout = 0.2
args.attention_type = 'local' # 'global', 'local'
args.align_type = 'general' # dot, general, concat, location
args.local_window = 10
args.input_feeding = True # True


## Learning ##
args.lr = 1.0
args.epoch = 10
args.batch_size = 128
args.lr_milestone = [6, 7, 8, 9, 10]
args.use_saved_vocab = False

## Options depend on dropout version
if args.dropout > 0.0:
    args.epoch = 12
    args.lr_milestone = [8, 9, 10, 11, 12]

## data prepareation:

input_vocab, output_vocab, train_pairs = prepare_data('train',args.max_len, False)
_, _, dev_pairs = prepare_data('dev',args.max_len,True)
_, _, test_pairs = prepare_data('test',args.max_len,True)

train_input_tensors, train_output_tensors,train_source_len_tensors= tensor_from_pair(train_pairs, input_vocab, output_vocab, args.max_len, "train")
del train_pairs
dev_input_tensors, dev_output_tensors,dev_source_len_tensors = tensor_from_pair(dev_pairs, input_vocab, output_vocab,args.max_len, "dev")
del dev_pairs
test_input_tensors, test_output_tensors,test_source_len_tensors = tensor_from_pair(test_pairs, input_vocab, output_vocab,args.max_len, "test")
del test_pairs

######################
train_input_tensors  = train_input_tensors.flip(1)
dev_input_tensors  = dev_input_tensors.flip(1)
test_input_tensors = test_input_tensors.flip(1)

######################
# shuffle=False -> 난수 생성과는 상관없이 매번 동일한 순서로 데이터를 입력
train_nmt_dataset = NMT_Dataset(train_input_tensors, train_output_tensors,train_source_len_tensors)
train_dataloader = DataLoader(train_nmt_dataset, batch_size = args.batch_size, shuffle = args.is_shuffle, drop_last=False)
del train_input_tensors, train_output_tensors

dev_nmt_dataset = NMT_Dataset(dev_input_tensors, dev_output_tensors,dev_source_len_tensors)
dev_dataloader = DataLoader(dev_nmt_dataset, batch_size = args.batch_size, shuffle = False, drop_last=False)
del dev_input_tensors, dev_output_tensors

test_nmt_dataset = NMT_Dataset(test_input_tensors, test_output_tensors,test_source_len_tensors)
test_dataloader = DataLoader(test_nmt_dataset, batch_size = args.batch_size, shuffle = False, drop_last=False)
del test_input_tensors, test_output_tensors


## Model
model = Seq2Seq(input_vocab.n_words, output_vocab.n_words ,args.hidden_dimension, args.hidden_dimension, args.batch_size, 
                args.device, args.n_layers, args.dropout, args.max_len,args.attention_type,args.align_type,args.input_feeding,args.local_window)
model.to(args.device)

## Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr)
scheduler = MultiStepLR(optimizer, milestones=args.lr_milestone, gamma=args.learning_rate_gamma)


## Training
train_losses = []
dev_losses = []
test_losses = []

train_ppls = []
dev_ppls = []
test_ppls = []

total_steps = 0
best_dev_ppl = float('inf')
best_dev_loss = float('inf')

for ep in range(args.epoch):

    model.train()

    train_loss = 0.0
    train_ppl = 0.0
    for i, (input_tensor, output_tensor,source_len_tensor) in enumerate(tqdm(train_dataloader, desc=f"Epoch {ep+1}", leave=False)):

        model.zero_grad()

        input_tensor = input_tensor.to(args.device)
        output_tensor = output_tensor.to(args.device)
        source_len_tensor = source_len_tensor.to(args.device)

        decoder_outputs, decoder_hidden = model(input_tensor,output_tensor,source_len_tensor)

        decoder_target = output_tensor
    
        test_loss += tloss.item()
        # Perplexity 계산
        t_perplexity = torch.exp(tloss)
        test_ppl += t_perplexity.item()
        
        ## view 연산은 메모리 공간상에 연속적인 공간을 가정한다.
        ## 하지만, padding 연산 같은 경우에는 메모리 공간상의 연속성을 보장하지 못하므로
        ## decoder_target에는 .contiguous 연산을 적용해줌
        ## decoder_outputs 같은 경우에는 Torch Tensor 연산의 결과물이므로
        ## 게산과정에서 연속 공간임이 보장됨

        # print(decoder_outputs.size()) # torch.Size([2, 51, 1325509])
        # print(decoder_target.size()) # torch.Size([2, 51])


        loss = args.criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), decoder_target.contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()

        train_loss += loss.item()
        # Perplexity 계산
        perplexity = torch.exp(loss)
        train_ppl += perplexity.item()
        
        total_steps += 1
        
        if total_steps % 5000 == 0:
            model.eval()
            
            test_loss = 0.0
            test_ppl = 0.0
            
            with torch.no_grad():
                for j, (test_input_tensors, test_output_tensor,test_source_len_tensor) in enumerate(test_dataloader):
                    
            
                    test_input_tensors = test_input_tensors.to(args.device)
                    test_output_tensor = test_output_tensor.to(args.device)
                    test_source_len_tensor = test_source_len_tensor.to(args.device)
                    test_decoder_outputs, test_decoder_hidden = model(test_input_tensors,test_output_tensor,test_source_len_tensor)

                    test_decoder_target = test_output_tensor

                    tloss = args.criterion(test_decoder_outputs.view(-1, test_decoder_outputs.size(-1)), test_decoder_target.contiguous().view(-1))
                    test_loss += tloss.item()
                    # Perplexity 계산
                    t_perplexity = torch.exp(tloss)
                    test_ppl += t_perplexity.item()
                    
                    
                test_loss = test_loss/len(test_dataloader)
                test_ppl = test_ppl/len(test_dataloader)
                                
                test_losses.append(test_loss)
                test_ppls.append(test_ppl)
                
                print(f"This is the test loss : {test_loss} ")
                print(f"This is the test  : {test_ppl} ")
            model.train()
                

    train_loss = train_loss/len(train_dataloader)
    train_ppl = train_ppl/len(train_dataloader)
    
    train_losses.append(train_loss)
    train_ppls.append(train_ppl)



    model.eval()
    dev_loss = 0.0
    dev_ppl = 0.0

    with torch.no_grad():
        for i, (input_tensor, output_tensor,dev_source_len_tensor) in enumerate(tqdm(dev_dataloader, desc="Dev", leave=False)):

            input_tensor = input_tensor.to(args.device)
            output_tensor = output_tensor.to(args.device)
            dev_source_len_tensor = dev_source_len_tensor.to(args.device)

            decoder_outputs, decoder_hidden = model(input_tensor,output_tensor,dev_source_len_tensor)

            decoder_target = output_tensor

            loss = args.criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), decoder_target.contiguous().view(-1))
            dev_loss += loss.item()

            # Perplexity 계산
            perplexity = torch.exp(loss)
            dev_ppl += perplexity.item()

        dev_loss = dev_loss/len(dev_dataloader)
        dev_ppl = dev_ppl/len(dev_dataloader)
        
        dev_losses.append(dev_loss)
        dev_ppls.append(dev_ppl)


    scheduler.step()

    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        torch.save(model.state_dict(), args.best_loss_model_path)
        print(f"{ep+1} 번째 epoch을 Best Loss Model로 저장합니다.")

    print(f"Iter: {i+1}, Epoch: {ep+1}, Loss(Train/Dev): {train_loss:.3f}/{dev_loss:.3f}",f"Ppl(Train/Dev): {train_ppl:.3f}/{dev_ppl:.3f}")

print(f"Total Learning is finalized")
print(f"this is the sequence of the test lossses : \n{test_losses}")
print(f"this is the sequence of the test ppls : \n{test_ppls}")


# 리스트를 딕셔너리로 묶기
results = {
    'test_losses': test_losses,
    'test_ppls': test_ppls
}

# results를 JSON 형식으로 파일에 저장
with open('results.txt', 'w') as f:
    json.dump(results, f)
    
    
# # 파일에서 JSON 형식의 데이터를 읽어오기
# with open('results.txt', 'r') as f:
#     results = json.load(f)

# # 딕셔너리에서 리스트 추출
# test_losses = results['test_losses']
# test_ppls = results['test_ppls']