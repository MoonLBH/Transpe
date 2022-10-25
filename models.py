from layers import *

class HalfTransformer(nn.Module):
    def __init__(self, d_model, d_ff, N, heads, dropout, activation) -> None:
        super(HalfTransformer,self).__init__()

        self.encoder = Encoder(d_model, d_ff, N, heads, dropout)
        self.predict = Linear_predictor(d_model, heads, activation, dropout)

    def forward(self,enc_input, enc_mask):
        enc_output = self.encoder(enc_input, enc_mask)
        output = self.predict(enc_output)
        return output

class CNNTransformer(nn.Module):
    def __init__(self, d_model, d_ff, N, heads, dropout, activation, oc1, oc2, hl) -> None:
        super(CNNTransformer,self).__init__()

        self.encoder = Encoder(d_model, d_ff, N, heads, dropout)
        self.predict = CNN_predictor(d_model, heads, activation, dropout,oc1,oc2,hl)

    def forward(self,enc_input, enc_mask):
        enc_output = self.encoder(enc_input, enc_mask)
        output = self.predict(enc_output)
        return output

class Linear_predictor(nn.Module):
    def __init__(self, d_model, heads, activation, dropout) -> None:
        super(Linear_predictor,self).__init__()
    
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(47* d_model* heads, d_model * heads),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_model * heads, d_model),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    
    def forward(self,input):
        output = self.fc(input)
        return output

class CNN_predictor(nn.Module):
    def __init__(self, d_model, heads, activation, dropout,oc1,oc2,hl) -> None:
        super(CNN_predictor,self).__init__()

        self.cnn_1 = nn.Sequential(
            nn.Conv1d(in_channels=d_model*heads, out_channels=oc1, kernel_size = 4),
            activation,
            nn.AvgPool1d(kernel_size = 2),
            nn.Dropout(dropout)
            )
        self.cnn_2 = nn.Sequential(
            nn.Conv1d(in_channels=oc1, out_channels=oc2, kernel_size = 3),
            activation,
            nn.AvgPool1d(kernel_size = 2),
            nn.Dropout(dropout)
            )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * oc2, hl),
            nn.Dropout(dropout),
            nn.Linear(hl,1)
        )
    def forward(self,input):
        input = input.permute(0,2,1)
        output = self.cnn_1(input)
        output = self.cnn_2(output)
        output = self.fc(output)
        return output
