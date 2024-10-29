import torch
import torch.nn as nn

class LSTMClassifier( nn.Module ):
    def __init__( self, nInputs,nHidden,nLayers,nOutput ):
        super( LSTMClassifier,self ).__init__( )
        self.D  = nInputs
        self.M  = nHidden
        self.K  = nOutput
        self.L  = nLayers

        self.lstm           = nn.LSTM(
                                            input_size      =self.D,
                                            hidden_size     =self.M,
                                            num_layers      =self.L,
                                            batch_first     =True
                                        )
        self.fc             = nn.Linear( self.M,self.K )
        self.softmax        = nn.Softmax( dim=1 )

    def forward( self,x ):
        out, (hn, cn)       = self.lstm( x )
        out                 = self.fc( hn[ -1 ] )
        return self.softmax( out )