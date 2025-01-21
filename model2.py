import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.weight_ih = nn.ParameterList([
            nn.Parameter(torch.Tensor(4 * hidden_size, input_size if i == 0 else hidden_size))
            for i in range(num_layers)
        ])
        self.weight_hh = nn.ParameterList([
            nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size)) for i in range(num_layers)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.Tensor(4 * hidden_size)) for i in range(num_layers)
        ])
        self.init_weights()

    def forward(self, x, hidden=None):
        is_packed = isinstance(x, PackedSequence)  # PackedSequence 여부 확인
        if is_packed:
            data, batch_sizes, sorted_indices, unsorted_indices = x
            batch_size = batch_sizes[0].item()
        else:
            data = x
            batch_size = x.size(0)

        device = data.device

        if hidden is None:
            h, c = self.init_hidden(batch_size, device)
        else:
            h, c = hidden

        if is_packed:
            outputs, (h, c) = self._packed_forward(data, batch_sizes, h, c)
            return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices), (h, c)
        else:
            outputs, (h, c) = self._unpacked_forward(data, h, c)
            return outputs, (h, c)

    def _packed_forward(self, data, batch_sizes, h, c):
        outputs = []
        split_data = data.split(batch_sizes.tolist())  # batch_sizes에 따라 데이터를 분할

        for t, batch_size in enumerate(batch_sizes):
            input_t = split_data[t]

            for layer in range(self.num_layers):
                h_t = h[layer][:batch_size].clone()  # 명시적으로 새로운 텐서 생성
                c_t = c[layer][:batch_size].clone()  # 명시적으로 새로운 텐서 생성  
                gates = (input_t @ self.weight_ih[layer].t() +
                         h_t @ self.weight_hh[layer].t() + self.bias[layer])
                
                i, f, o, g = gates.chunk(4, 1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                o = torch.sigmoid(o)
                g = torch.tanh(g)

                new_c_t = f * c_t + i * g         
                new_h_t = o * torch.tanh(new_c_t)
                new_h_t, new_c_t = self.clip_states(new_h_t, new_c_t, 50)
                input_t = new_h_t

                
                ## 특정 그래프가 깨질 경우 아래와 같이 grad_fn 없이 출력되면 해당 부분이 문제가 됨
                
                # f: tensor([[0.3, 0.5, ...]], grad_fn=<SigmoidBackward>)
                # c_t: tensor([[0.2, 0.7, ...]])  # <--- grad_fn 없음
                # i: tensor([[0.6, 0.4, ...]], grad_fn=<SigmoidBackward>)
                # g: tensor([[0.1, -0.2, ...]], grad_fn=<TanhBackward>)
                # new_c_t: tensor([[0.5, 0.9, ...]], grad_fn=<AddBackward>)
                # 이 부분은 inplace 방식이라고 함
                # inplcae 방식을 사용해서 수정하면 계산그래프가 올바르게 동작하지 않는다고 함
                ## inplace 방식이란 tensor를 직접 수정하는 방식
                ## 텐서를 슬라이싱 하고 슬라이싱 된 부분에 새로운 값을 넣는것임
                ## 계산그래프를 추적할 때 문제가 됨
                ## 역전파를 위해서 순전파떄의 정보가 남아있어야 함
                # h[layer][:batch_size] = h_t 
                # c[layer][:batch_size] = c_t

                # 새로운 텐서를 생성하여 업데이트
                # 새로운 텐서를 생성하여 업데이트

                # 새로운 Tensor로 교체
                



                h[layer] = torch.cat([new_h_t, h[layer][batch_size:]], dim=0)
                c[layer] = torch.cat([new_c_t, c[layer][batch_size:]], dim=0)
                            
            outputs.append(input_t)

        outputs = torch.cat(outputs, dim=0)  # 분할된 데이터를 다시 병합
        return outputs, (h, c)
    
    def _unpacked_forward(self, data, h, c):
        outputs = []
        seq_len = data.size(1)  # sequence length

        for t in range(seq_len):
            input_t = data[:, t, :]  # 현재 타임스텝의 입력

            for layer in range(self.num_layers):
                h_t = h[layer].clone()  # clone()으로 복사하여 수정
                c_t = c[layer].clone()  # clone()으로 복사하여 수정

                gates = (input_t @ self.weight_ih[layer].t() +
                        h_t @ self.weight_hh[layer].t() + self.bias[layer])
                
                i, f, o, g = gates.chunk(4, 1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                o = torch.sigmoid(o)
                g = torch.tanh(g)
                
                # 새로운 텐서 생성
                new_c_t = f * c_t + i * g
                new_h_t = o * torch.tanh(new_c_t)
                
                new_h_t, new_c_t = self.clip_states(new_h_t, new_c_t, 50)
                input_t = new_h_t

                # 상태 업데이트
                h[layer] = new_h_t.clone() # detach()로 그래프 분리
                c[layer] = new_c_t.clone()  # detach()로 그래프 분리

            outputs.append(input_t.unsqueeze(1))  # 타임스텝 차원을 추가하여 저장

        outputs = torch.cat(outputs, dim=1)  # 타임스텝 차원을 따라 병합
        return outputs, (h, c)

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def init_weights(self):
        for weight in self.weight_ih:
            nn.init.uniform_(weight, -0.1, 0.1)
        for weight in self.weight_hh:
            nn.init.uniform_(weight, -0.1, 0.1)
        for bias in self.bias:
            nn.init.zeros_(bias)
    # Python으로 구현한 클리핑 함수
    def clip_states(self, h_t, c_t, clip_value):
        # c_t와 h_t의 값을 clip_value 범위로 제한
        c_t = torch.clamp(c_t, min=-clip_value, max=clip_value)
        h_t = torch.clamp(h_t, min=-clip_value, max=clip_value)
        return h_t, c_t
                
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_size, hidden_size, batch_size, device, n_layers=4, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_size, padding_idx = 2)
        #self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=n_layers,batch_first = True, dropout =dropout)
        self.lstm = CustomLSTM(embedding_size, hidden_size, num_layers=n_layers)

    def forward(self, x): # x : torch.Size([128, 50]) (batch_size, seq_len)
        # 패딩이 아닌 시퀀스 길이 계산
        lengths = (x != 2).sum(dim=1)  # input_tensor: [batch_size, seq_len]
        
        embed_x = self.embedding(x) # embed_x : torch.Size([128, 50, 1000]) (batch_size, seq_len, embedding_size)
        
        # GPU 텐서를 CPU로 변환
        lengths_cpu = lengths.cpu()
        
        # # Packed Sequence 생성
        packed_input = pack_padded_sequence(embed_x, lengths_cpu, batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)

        # hidden = torch.clamp(hidden, min=-50, max=50)
        # cell = torch.clamp(cell, min=-50, max=50)

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
        

class Decoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_size, hidden_size, batch_size, device, n_layers=4, dropout=0.2, max_len=50):
        super().__init__()
        # Embedding
        self.src_vocab_size = src_vocab_size
        self.embedding = nn.Embedding(self.src_vocab_size, embedding_size, padding_idx = 2)
        # LSTM
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=n_layers,batch_first = True, dropout =dropout)
        #self.lstm = CustomLSTM(embedding_size, hidden_size, num_layers=n_layers)

        #self.init_cell = torch.zeros(n_layers, batch_size, hidden_size).to(device)
        #nn.init.uniform_(self.init_cell, -0.1, 0.1)
        # Linear
        self.out = nn.Linear(hidden_size, self.src_vocab_size)

        self.teacher_forcing_ratio = 1
        self.max_len = max_len
        self.SOS_token = 0
        self.device = device
    def forward(self, encoder_output, decoder_hidden, decoder_cell, target_tensor): # encoder_output torch.Size([128, 50, 1000]) target [2698, 50]

        batch_size = decoder_hidden.size(1) # decoder_hidden torch.Size([4, 128, 1000])

        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_outputs = []
        decoder_outputs = torch.zeros(batch_size, self.max_len+1, self.src_vocab_size, device=self.device)

        for i in range(self.max_len+1): # 0~50 SOS 들어가고 애초에 마지막 입력은 for문 안돌아도 됨
            decoder_output, decoder_hidden, decoder_cell = self.forward_step(decoder_input, decoder_hidden, decoder_cell)
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

    def forward_step(self, decoder_input,decoder_hidden,decoder_cell):

        embed_x = self.embedding(decoder_input) # print(embed_x.size()) (batch=128, 1, 1000)

        decoder_output, (decoder_hidden,decoder_cell) = self.lstm(embed_x, (decoder_hidden, decoder_cell))
        decoder_hidden = torch.clamp(decoder_hidden, min=-50, max=50)
        decoder_cell = torch.clamp(decoder_cell, min=-50, max=50)
        
        decoder_output  = self.out(decoder_output) # 128, 1, 2XXXX

        return decoder_output, decoder_hidden, decoder_cell

    def initialization(self):
        nn.init.uniform_(self.embedding.weight,-0.1, 0.1)
        nn.init.uniform_(self.out.weight,-0.1, 0.1)
        nn.init.constant_(self.out.bias, 0.0)
        

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, ref_vocab_size ,embedding_size, hidden_size, batch_size,  device, n_layers=4, dropout=0.2, max_len = 50):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embedding_size, hidden_size, batch_size, device, n_layers, dropout)
        self.decoder = Decoder(ref_vocab_size, embedding_size, hidden_size, batch_size, device, n_layers, dropout, max_len)

        self.initialization()

    def forward(self, input_tensor, output_tensor):


        encoder_outputs, encoder_hidden, encoeder_cell = self.encoder(input_tensor)
        decoder_hidden = encoder_hidden
        
        # print(decoder_hidden.shape) # torch.Size([4, 128, 1000])
        # print(encoeder_cell.shape) # torch.Size([4, 128, 1000])
        decoder_outputs, decoder_hidden, _ = self.decoder(encoder_outputs, encoder_hidden, encoeder_cell, output_tensor)

        return decoder_outputs, decoder_hidden
    
    def initialization(self):
        self.encoder.initialization()
        self.decoder.initialization()