import torch
import torch.nn as nn
import numpy as np

class TimeSeriesTransformer( nn.Module ):

    def __init__( self, nInputs,dModel,nHead,nLayers,dimFeedForward,nOutput,dropoutRate=0.1 ):
        super( TimeSeriesTransformer,self ).__init__( )
        self.D                  = nInputs
        self.dModel             = dModel
        self.inputProjection    = nn.Linear(  nInputs, dModel  )
        self.posEncoding        = nn.Parameter( self._generatePosEncoding( dModel, maxLength=500 ), requires_grad=False )
        encoderLayer            = nn.TransformerEncoderLayer( d_model=dModel, nhead=nHead, dim_feedforward=dimFeedForward, dropout=dropoutRate )
        self.transformerEncoder = nn.TransformerEncoder(encoderLayer, num_layers=nLayers)
        self.fc                 = nn.Linear( dModel, nOutput  )
        self.softmax            = nn.Softmax( dim=1 )

    def _generatePosEncoding(  self, dModel, maxLength  ):
        position                = torch.arange(  0, maxLength  ).unsqueeze(1)
        divTerm                 = torch.exp(torch.arange(  0, dModel, 2) * -(np.log(10000.0) / dModel)  )
        posEncoding             = torch.zeros(  maxLength, dModel  )
        posEncoding[ :, 0::2 ]    = torch.sin(  position * divTerm  )
        posEncoding[ :, 1::2 ]    = torch.cos(  position * divTerm  )
        return posEncoding.unsqueeze( 0 )

    def forward( self, x ):
        x                       = self.inputProjection( x )
        x                       = x + self.posEncoding[ :, :x.size(1), : ]
        x                       = x.permute(1, 0, 2)
        x                       = self.transformerEncoder( x )
        x                       = x[ -1, :, : ]
        output                  = self.fc( x )
        return self.softmax( output )
    

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

    def forward( self,x ):
        out, (hn, cn)       = self.lstm( x )
        out                 = self.fc( hn[ -1 ] )
        return out