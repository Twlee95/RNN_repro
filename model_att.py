import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_size, hidden_size, batch_size, device, n_layers=4, dropout=0.2,max_len=50, attention_type=None):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_size, padding_idx = 2)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=n_layers,batch_first = True, dropout =dropout)
        self.atttention_type = attention_type
        # self.lstm = CustomLSTM(embedding_size, hidden_size, num_layers=n_layers)
        self.max_len = max_len
    def forward(self, x): # x : torch.Size([128, 50]) (batch_size, seq_len)
        # 패딩이 아닌 시퀀스 길이 계산
        #lengths = (x != 2).sum(dim=1)  # input_tensor: [batch_size, seq_len]
        
        embed_x = self.embedding(x) # embed_x : torch.Size([128, 50, 1000]) (batch_size, seq_len, embedding_size)
        
        # GPU 텐서를 CPU로 변환
        #lengths_cpu = lengths.cpu()
        
        # # Packed Sequence 생성
        #packed_input = pack_padded_sequence(embed_x, lengths_cpu, batch_first=True, enforce_sorted=False)

        outputs, (hidden, cell) = self.lstm(embed_x)
        
        #outputs, _ = pad_packed_sequence(packed_output, batch_first=True, total_length= self.max_len)

        hidden = torch.clamp(hidden, min=-50, max=50)
        cell = torch.clamp(cell, min=-50, max=50)

        # print("outputs",outputs.size()) # outputs torch.Size([128, 50, 2000])
        # print("hidden",hidden.size()) # hidden torch.Size([4, 128, 1000])
        # print("cell",cell.size()) # cell torch.Size([4, 128, 1000])

        return outputs, hidden, cell

    def initialization(self):
        nn.init.uniform_(self.embedding.weight,-0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param,-0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)   
        
class Alignemnt_Predictor(nn.Module):
    def __init__(self, hidden_size, max_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.W_p = nn.Linear(hidden_size,hidden_size,bias = False)
        self.V_p = nn.Linear(hidden_size,1,bias = False)
        nn.init.uniform_(self.W_p.weight,-0.1, 0.1)
        nn.init.uniform_(self.V_p.weight,-0.1, 0.1)

        
    def forward(self, decoder_hidden,source_len): #[128,1,1000]
        pre_vec = torch.tanh(self.W_p(decoder_hidden)) #[128,1,1000]
        pre_scalar = self.V_p(pre_vec).squeeze() #[128]
        
        # print("source_len",source_len.size())
        # print("pre_scalar",pre_scalar.size())
        return source_len*torch.sigmoid(pre_scalar)


class Attention(nn.Module):
    def __init__(self, hidden_size,max_len,attn_type,align_type,device,local_window=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.device = device
        
        ## local
        self.local_linear = nn.Linear(hidden_size,hidden_size, bias=False)
        
        ## global
        self.global_linear = nn.Linear(hidden_size,50, bias=False)
        self.softmax = nn.Softmax(dim=1)
        
        self.out_linear = nn.Linear(2*hidden_size,hidden_size, bias=False)
        self.attn_type = attn_type
        self.align_type = align_type
        self.local_window = local_window
        
        nn.init.uniform_(self.local_linear.weight,-0.1, 0.1)
        nn.init.uniform_(self.global_linear.weight,-0.1, 0.1)
        nn.init.uniform_(self.out_linear.weight,-0.1, 0.1)
                
    def forward(self, encoder_output, decoder_output,p_t=None): # encoder_output 128, 50, 1000  # decoder_output 128, 1, 1000 
        batch_size = encoder_output.size(0)
        # print(decoder_output.size(),"decoder_output") torch.Size([128, 1, 1000])
        align_fn = None
        # size limit
        if self.attn_type == 'local':
            start_idx = 0
            end_idx = self.max_len

            p_t = self.max_len - p_t
            
            # set index
            left_idx = p_t - self.local_window # tensor - scalar
            right_idx = p_t + self.local_window
            left_over = left_idx < start_idx
            left_idx[left_over] = start_idx
            right_over = right_idx > end_idx
            right_idx[right_over] = end_idx

            ## general score function
            lin_decodeer = self.local_linear(decoder_output) # [128, 1, 1000]
            score_fn = torch.bmm(lin_decodeer, encoder_output.transpose(1,2)).squeeze(1) # [128, 1, 50]

            
            ##  masking
            seq_range = torch.arange(self.max_len).unsqueeze(0).expand(batch_size, -1).to(self.device)  # [batch_size, seq_len]
            mask = (seq_range < left_idx.unsqueeze(-1)) | (seq_range > right_idx.unsqueeze(-1))
            # if mask.all():
            #     score_fn = torch.full_like(score_fn, -1e9)  # 모든 값을 -1e9로 설정
            # else:
            #     score_fn.masked_fill_(mask, -float('inf'))  # 마스킹된 부분 -inf로 설정
            # score_fn = score_fn - score_fn.max(dim=1, keepdim=True).values
            

            #score_fn = score_fn - score_fn.max(dim=1, keepdim=True).values
            score_fn.masked_fill_(mask, -float('inf'))
            align_fn = score_fn.softmax(dim=1)
            
            ## gaussian
            gaussian_idx = torch.arange(self.max_len).unsqueeze(0).expand(batch_size, -1).to(self.device)
            p_t_r = p_t.unsqueeze(-1)
            #gaussian_norm_matrix = torch.exp(-((gaussian_idx-p_t_r)**2)/(2*((self.local_window/2)**2)))
            gaussian_norm_matrix = torch.exp(-((gaussian_idx - p_t_r)**2).clamp(max=1e2) / (2 * ((self.local_window / 2)**2)))           
            att_score = align_fn * gaussian_norm_matrix # 128,50
 
        if self.attn_type == 'global':            
            global_linear = self.global_linear(decoder_output).squeeze(1)  # g_l : torch.Size([128, 50])
            att_score = self.softmax(global_linear) # a_s [128,50]

        score_vector = torch.mul(encoder_output,att_score.unsqueeze(2)) # [128,50,1000]
        attention_vector = torch.sum(score_vector, dim = 1) # [128,1000]
        concat_vec = torch.cat((decoder_output.squeeze(1),attention_vector),dim = 1) # [128,2000]
        attention_output =  self.out_linear(concat_vec).unsqueeze(1) #[128,1,1000]
        tanh_attention_output = torch.tanh(attention_output) 
    
        return tanh_attention_output,align_fn #[128,1,1000]
    
    
    
    # def forward(self, encoder_output, decoder_output): # encoder_output 128, 50, 1000  # decoder_output 128, 1, 1000 
    #     enc_seq_len = encoder_output.size(1)
    #     # print(decoder_output.size(),"decoder_output") torch.Size([128, 1, 1000])

    #     global_linear = self.global_linear(decoder_output)  # g_l : torch.Size([128, 1, 50])
        
    #     att_score = self.softmax(global_linear) # a_s [128,1,50]

    #     attention_vector = torch.bmm(att_score, encoder_output) # [128,1,50] [128,50,1000] =[128, 1, 1000]
        
    #     concat_vev = torch.cat((decoder_output, attention_vector), dim = 2) # [128, 1, 2000]
        
    #     attention_output =  self.out_linear(concat_vev) #[128,1,1000]
        
    #     return torch.tanh(attention_output) #[128,1,1000]
        


class Decoder(nn.Module):
    
    def __init__(self, src_vocab_size, embedding_size, hidden_size, batch_size, device, n_layers=4, dropout=0.2, max_len=50, 
                 attention_type=None,align_type=None,input_feeding=False,local_window=None):
        super().__init__()
        # Embedding
        self.src_vocab_size = src_vocab_size
        self.embedding = nn.Embedding(self.src_vocab_size, embedding_size, padding_idx = 2)
        # LSTM
        if input_feeding == False:
            self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=n_layers,batch_first = True, dropout =dropout)
        else:
            self.lstm = nn.LSTM(embedding_size+hidden_size, hidden_size, num_layers=n_layers,batch_first = True, dropout =dropout)

        #self.lstm = CustomLSTM(embedding_size, hidden_size, num_layers=n_layers)

        #self.init_cell = torch.zeros(n_layers, batch_size, hidden_size).to(device)
        #nn.init.uniform_(self.init_cell, -0.1, 0.1)
        # Linear
        self.out = nn.Linear(hidden_size, self.src_vocab_size)

        self.teacher_forcing_ratio = 1
        self.max_len = max_len
        self.SOS_token = 0
        self.device = device
        self.attention_type = attention_type
        self.align_type = align_type
        self.input_feeding = input_feeding
        self.hidden_size = hidden_size
        self.local_window = local_window 
        
        if self.attention_type == 'global':
            self.attention = Attention(hidden_size,max_len,attention_type,align_type,device)
        
        if self.attention_type == 'local':
            self.align_predictor= Alignemnt_Predictor(hidden_size,max_len)
            self.attention = Attention(hidden_size,max_len,attention_type,align_type,device,local_window)
        
    def forward(self, encoder_output, decoder_hidden, decoder_cell, target_tensor,source_len): # encoder_output torch.Size([128, 50, 1000]) target [2698, 50]

        batch_size = decoder_hidden.size(1) # decoder_hidden torch.Size([4, 128, 1000])

        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_outputs = []
        decoder_outputs = torch.zeros(batch_size, self.max_len+1, self.src_vocab_size, device=self.device)
        attention_output = torch.zeros(batch_size, 1, self.hidden_size, device=self.device)
        
        for i in range(self.max_len+1): # 0~50 SOS 들어가고 애초에 마지막 입력은 for문 안돌아도 됨
            decoder_output, decoder_hidden, decoder_cell, attention_output, align_fn = self.forward_step(encoder_output,decoder_input, decoder_hidden, decoder_cell,attention_output,source_len)
            decoder_outputs[:, i, :] = decoder_output.squeeze(1)
            #decoder_input = target_tensor[:,i].unsqueeze(1)
            #     decoder_outputs[:, i, :] = decoder_output.squeeze(1)
            # IndexError: index 51 is out of bounds for dimension 1 with size 51

            # Teacher forcing 적용
            use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # 타겟 텐서를 다음 입력으로 사용
            else:
                decoder_input = decoder_output.argmax(dim=-1)  # 예측된 출력을 다음 입력으로 사용
                
            #print("decoder_output", decoder_output.size()) # decoder_output torch.Size([128, 1, 12059])

        # print("decoder_outputs",decoder_outputs.size()) # # decoder_outputs torch.Size([128, 51, 12059])
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs, decoder_hidden , None

    def forward_step(self, encoder_output, decoder_input,decoder_hidden,decoder_cell,attention_output,source_len):

        embed_x = self.embedding(decoder_input) # print(embed_x.size()) (batch=128, 1, 1000)
        
        if self.input_feeding == True:
            embed_x = torch.cat((embed_x, attention_output), dim = 2)
        
        decoder_output, (decoder_hidden,decoder_cell) = self.lstm(embed_x, (decoder_hidden, decoder_cell))
        decoder_hidden = torch.clamp(decoder_hidden, min=-50, max=50)
        decoder_cell = torch.clamp(decoder_cell, min=-50, max=50)
        
        #print("decoder_hidden",decoder_hidden.size()) torch.Size([4, 128, 1000])

        if self.attention_type == "global":
            attention_output, _ = self.attention(encoder_output,decoder_output) #[128,1,1000]
        if self.attention_type == 'local':
            p_t = self.align_predictor(decoder_output,source_len)
            attention_output,align_fn = self.attention(encoder_output,decoder_output,p_t)

        if self.attention_type == None: # attention
            decoder_output  = self.out(decoder_output) # 128, 1, 2XXXX
        else:
            decoder_output  = self.out(attention_output) # 128, 1, 2XXXX

        return decoder_output, decoder_hidden, decoder_cell, attention_output, None

    def initialization(self):
        nn.init.uniform_(self.embedding.weight,-0.1, 0.1)
        nn.init.uniform_(self.out.weight,-0.1, 0.1)
        nn.init.constant_(self.out.bias, 0.0)
        

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, ref_vocab_size ,embedding_size, hidden_size, batch_size,  device, n_layers=4, dropout=0.2, 
                 max_len = 50,attention_type=None,alignment_type=None,input_feeding=False,local_window=None):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embedding_size, hidden_size, batch_size, device, n_layers, dropout,max_len,attention_type)
        self.decoder = Decoder(ref_vocab_size, embedding_size, hidden_size, batch_size, device, n_layers, dropout, max_len,attention_type,alignment_type,input_feeding,local_window)

        self.initialization()

    def forward(self, input_tensor, output_tensor,source_len):


        encoder_outputs, encoder_hidden, encoeder_cell = self.encoder(input_tensor)
        decoder_hidden = encoder_hidden
        
        # print(decoder_hidden.shape) # torch.Size([4, 128, 1000])
        # print(encoeder_cell.shape) # torch.Size([4, 128, 1000])
        decoder_outputs, decoder_hidden, _ = self.decoder(encoder_outputs, encoder_hidden, encoeder_cell, output_tensor,source_len)

        return decoder_outputs, decoder_hidden
    
    def initialization(self):
        self.encoder.initialization()
        self.decoder.initialization()