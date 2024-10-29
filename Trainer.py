import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import scipy as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from LoggerDev import LoggerDev
import os
from Nomenclature import Nomenclature
import sys
from Transformer import TimeSeriesTransformer
from LSTM import LSTMClassifier
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix

class Trainer:

    FDICT_PLOTS                 = {'family':'Arial','size':8}

    def __init__( self ):
        self.logger             = LoggerDev(  )
        self.dataSets           = [  ]
        self.blockWeights       = [  ]
        self.nom                = Nomenclature(  )
        self.labelPreds         = {  }

    def loadData( self,pathFolder,blockWeights={  } ):
        for file in os.listdir( pathFolder ):
            df                  = pd.read_csv( pathFolder+"\\"+file,parse_dates=[ self.nom.DATE_MNEMONIC ] )
            if file[ :-4 ] in list( blockWeights.keys(  ) ):    
                bW              =   blockWeights[ file[ :-4 ] ]
            else:                                               
                bW              =   self._getBlockWeight( df )
                self.logger.warningMsg( f"The block weight for the {file[ :-4 ]} well was not provided. Therefore, it has been inferred. Its value is: {bW}" )
            self.dataSets.append( df )
            self.blockWeights.append( bW )
        self.logger.infoMsg( f"Data has been loaded correctly to the trainer. In total, {len( self.dataSets )} dataframes have been loaded." )

    def trainModel(  self, modelType, batchSize=32, nEpochs=1000, learningRate=0.0001, currentCheckPointPath=None, 
                    saveModel=True, savePath="model_.pth", **kwargs  ):
        if len( self.dataSets )==0:
            self.logger.errorMsg( "No data has been loaded to the trainer. Use the loadData( ) function before training a model." )
            sys.exit( 1 );
        model, optimizer        = self._createModel( modelType,learningRate, **kwargs )
        currEpoch               = 0
        trainLosses             = np.zeros( nEpochs*len( self.dataSets ) )
        if currentCheckPointPath is not None:   
            model, optimizer    = self._loadCheckPoint( currentCheckPointPath, model, optimizer )
            self.logger.infoMsg( "Checkpoint loaded successfully. Resuming training..." )
        else:                   self.logger.infoMsg( "Starting training..." )
        for k,ds in enumerate( self.dataSets ):
            X_train, y_train, _     = self.extractFeatures( ds,self.blockWeights[ k ],modelType=modelType )
            criterion               = nn.CrossEntropyLoss(  )        
            dataset                 = TensorDataset( X_train, y_train )
            dataLoader              = DataLoader( dataset, batch_size=batchSize, shuffle=True )
            for epoch in range( nEpochs ):
                if epoch==0: self.logger.infoMsg( f"Initiating forward pass (epoch {epoch}) for dataset {k}." )
                model.train(  )
                runningLoss         = 0.0
                correctPreds        = 0
                totalPreds          = 0
                for inputs, labels in dataLoader:
                    optimizer.zero_grad(  )
                    outputs         = model( inputs )
                    loss            = criterion( outputs, labels )
                    loss.backward(  )
                    optimizer.step(  )
                    runningLoss     += loss.item(  )
                    _, predicted    = torch.max( outputs, 1 )
                    correctPreds    += ( predicted == labels ).sum(  ).item(  )
                    totalPreds      += labels.size( 0 )
                epochLoss               = runningLoss / len(  dataLoader  )
                trainLosses[ k*nEpochs+epoch ] = epochLoss
                if ( epoch + 1 ) % 2 == 0:
                     self.logger.infoMsg( f'[TRAINING MSG>>>]..... Epoch {epoch+1}/{nEpochs}, Train Loss: {loss.item(  ):.4f}')
            self._saveCheckpoint( model,optimizer,epoch,trainLosses[-1],modelType,codeName=f"_{k}" )
            self.logger.infoMsg( f"Successfully saved checkpoint: {modelType}_{k}_chk" )
        if saveModel:   
            torch.save(  model.state_dict(  ), savePath  )
            self.logger.infoMsg( f"Successfully saved {modelType} model: {savePath}" )
        return trainLosses, model

    def _loadCheckpoint( self, currentCheckPointPath, model, optimizer ):
        checkpoint          = torch.load(  currentCheckPointPath, weights_only=True  )
        model.load_state_dict(  checkpoint['model_state_dict']  )
        optimizer.load_state_dict(  checkpoint['optimizer_state_dict']  )
        return model, optimizer

    def _getBlockWeight( self,dataSet ):
        return dataSet[ dataSet[ self.nom.HOOK_LOAD_MNEMO ]>0.0][ self.nom.HOOK_LOAD_MNEMO ].mode(  )[ 0 ]

    def extractFeatures( self, df, blockWeight, modelType,slidingWindow = 5 ):
        device          = torch.device( "cuda:0" if torch.cuda.is_available( ) else "cpu" )
        typeDevice      = "GPU" if torch.cuda.is_available( ) else "CPU"
        self.logger.infoMsg(f"Working on: {typeDevice}, model {torch.cuda.get_device_name(0)}")
        df[self.nom.EFF_HOOK_LOAD_MNEMO]            = np.where(df[self.nom.HOOK_LOAD_MNEMO]>0,df[self.nom.HOOK_LOAD_MNEMO]-blockWeight,df[self.nom.HOOK_LOAD_MNEMO])
        df[self.nom.BLOCK_POSITION_TREND_MNEMO]     = df[self.nom.BLOCK_POSITION_MNEMO].rolling(window=slidingWindow).apply(self._trend,raw=True,engine='cython')
        df[self.nom.FLOW_RATE_VARIABILITY_MNEMO]    = df[self.nom.FLOW_IN_MNEMO].rolling(window=slidingWindow).std(  )
        df[self.nom.FLOW_RATE_MEAN_MNEMO]           = df[self.nom.FLOW_IN_MNEMO].rolling(window=slidingWindow).mean(  )
        df[self.nom.RPM_MEAN_MNEMO]                 = df[self.nom.RPM_MNEMO].rolling(window=slidingWindow).mean(  )
        df[self.nom.HOOK_LOAD_MEAN_MNEMO]           = df[self.nom.EFF_HOOK_LOAD_MNEMO].rolling(window=slidingWindow).mean(  )
        df[self.nom.HOOK_LOAD_VARIABILITY_MNEMO]    = df[self.nom.EFF_HOOK_LOAD_MNEMO].rolling(window=slidingWindow).std(  )
        df[self.nom.ROP_MEAN_MNEMO]                 = df[self.nom.ROP_MNEMO].rolling(window=slidingWindow).mean(  )
        dfTraining                                  = df[[self.nom.BLOCK_POSITION_TREND_MNEMO,
                                                            self.nom.FLOW_RATE_VARIABILITY_MNEMO,
                                                            self.nom.FLOW_RATE_MEAN_MNEMO,
                                                            self.nom.RPM_MEAN_MNEMO,
                                                            self.nom.HOOK_LOAD_MEAN_MNEMO,
                                                            self.nom.ROP_MEAN_MNEMO,
                                                            self.nom.HOOK_LOAD_VARIABILITY_MNEMO,
                                                            self.nom.RIG_STATE_MNEMO]]
        dfTraining                                  = dfTraining.dropna( axis=0 )
        X                       = dfTraining.iloc[:,:-1].values
        y                       = df[self.nom.RIG_STATE_MNEMO].values
        y                       = pd.get_dummies( y )
        y                       = y.reindex( columns=self.nom.GOAL_RIG_STATES,fill_value=False )
        y                       = y.values
        scaler                  = StandardScaler( )
        XSc                     = scaler.fit_transform( X )
        XF                      = [ ]
        yF                      = [ ]
        for t in range( XSc.shape[ 0 ]-slidingWindow ):
            x                   = XSc[ t:t+slidingWindow,: ]
            XF.append( x )
            ny                  = y[ t+slidingWindow-1,: ]
            yF.append( ny )
        XF                      = np.array( XF ).reshape( -1,slidingWindow,XSc.shape[1] )
        yF                      = np.array( yF ).reshape( -1,y.shape[1] )
        X_train                 = torch.from_numpy(XF.astype(np.float32)).to( device )
        y_train                 = torch.from_numpy(yF.astype(np.float32))
        y_train                 = torch.argmax( y_train, dim=1 ).to( device )
        # if modelType=="lstm":
        #     y_train                 = torch.from_numpy(yF.astype(np.float32)).to( device )
        # else:
        #     y_train                 = torch.from_numpy(yF.astype(np.float32))
        #     y_train                 = torch.argmax( y_train, dim=1 ).to( device )
        return X_train,y_train,device
    
    def plotLosses( self, trainingLoss, testLoss, outPath="loss_curve.png" ):
        fig,ax                    = plt.subplots( figsize=(6.83,3.33) )
        ax.plot( np.arange( len(trainingLoss) ), trainingLoss, lw=1.5, color='teal',label="Training Loss" )
        if not testLoss is None: ax.plot( np.arange( len(testLoss) ), testLoss, lw=1.5, color='red',ls='--',label="Test Loss" )
        ax.set_xlabel( 'Epoch', fontdict={ **self.FDICT_PLOTS,'weight':'bold'} )
        ax.set_ylabel( 'Loss (Cross Entropy)', fontdict={ **self.FDICT_PLOTS,'weight':'bold'} )
        ax.set_xticklabels( ax.get_xticklabels( ), fontdict=self.FDICT_PLOTS )
        ax.set_yticklabels( ax.get_yticklabels( ), fontdict=self.FDICT_PLOTS )
        ax.yaxis.set_minor_locator( AutoMinorLocator( ) )
        ax.legend(  )
        plt.tight_layout(  )
        fig.savefig( outPath,dpi=600 )

    def _plotFancyContingencyTable(  self, confMatrix, classNames,outPath="contingency_table.png"  ):
        fig, ax                 = plt.subplots(  figsize=(  6.83, 4  ))
        buff                    = sb.heatmap(  confMatrix, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=classNames, yticklabels=classNames, ax=ax, annot_kws={ **self.FDICT_PLOTS }  )
        ax.set_xlabel(  "Predicted Labels",fontdict={ **self.FDICT_PLOTS, 'weight':'bold' }  )
        ax.set_ylabel(  "True Labels",fontdict={ **self.FDICT_PLOTS, 'weight':'bold' }  )
        ax.set_title(  "Confusion Matrix",fontdict={ **self.FDICT_PLOTS, 'weight':'bold' }  )
        ax.set_xticklabels( ax.get_xticklabels( ), fontdict=self.FDICT_PLOTS )
        ax.set_yticklabels( ax.get_yticklabels( ), fontdict=self.FDICT_PLOTS )      
        plt.tight_layout(  )
        fig.savefig( outPath,dpi=600 )  

    def _saveCheckpoint( self,model, optimizer, epoch, loss, modelType, path='checkpoint.pth',codeName="_" ):
        newPath    = f"{modelType}{codeName}_chk"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(  ),
            'optimizer_state_dict': optimizer.state_dict(  ),
            'loss': loss
        }
        torch.save(  checkpoint, newPath  )
        self.logger.infoMsg( f"Checkpoint saved at epoch {epoch}." )

    def _createModel( self,modelType,learningRate=0.0001,**kwargs ):
        device = torch.device( "cuda:0" if torch.cuda.is_available( ) else "cpu" )
        if modelType==self.nom.TRANSFORMER_MODEL_MNEMO:
            model                   = self.createModelTransformer( device, **kwargs )            
        elif modelType==self.nom.LSTM_MODEL_MNEMO:
            model                   = self.createModelLSTM( device, **kwargs )
        optimizer                   = torch.optim.Adam( model.parameters( ),lr=learningRate )
        return model,optimizer

    def createModelTransformer( self, device, **kwargs ):
        nInputs         = 7 if "nInputs" not in kwargs else kwargs["nInputs"]
        nLayers         = 3 if "nLayers" not in kwargs else kwargs["nLayers"]
        dModel          = 256 if "dModel" not in kwargs else kwargs["dModel"]
        nOutput         = 10 if "nOutput" not in kwargs else kwargs["nOutput"]
        nHead           = 8 if "nHead" not in kwargs else kwargs["nHead"]
        dimFeedForward  = 512 if "dimFeedForward" not in kwargs else kwargs["dimFeedForward"]
        model           = TimeSeriesTransformer(  nInputs=nInputs,nLayers=nLayers,dModel=dModel,
                                        nOutput=nOutput,nHead=nHead,dimFeedForward=dimFeedForward )
        model.to( device )
        return model

    def createModelLSTM( self,device, **kwargs ):
        nInputs         = 7 if "nInputs" not in kwargs else kwargs["nInputs"]
        nLayers         = 3 if "nLayers" not in kwargs else kwargs["nLayers"]
        nHidden         = 30 if "nHidden" not in kwargs else kwargs["nHidden"]
        nOutput         = 10 if "nOutput" not in kwargs else kwargs["nOutput"]
        model           = LSTMClassifier(  nInputs=nInputs,nLayers=nLayers, nOutput=nOutput,nHidden=nHidden )
        model.to( device )
        return model

    def _trend( self,window ):
        slope,_,_,_,_         = sp.stats.linregress( np.arange(len( window )), window )
        return slope

    def testModel( self,pathTest,blockWeights={  },batchSize=512,loadFromPath=None,modelType=None,model=None,**kwargs ):
        if (loadFromPath is None) & (model is None):
            self.logger.errorMsg( "Cannot test a model if no model is indicated. Either specify the torch model object or a path to a .pth file containing a model." )
            sys.exit( 1 )
        if loadFromPath is not None:
            if modelType is None:
                self.logger.errorMsg( "The type of model must be specified. Options: 'lstm', 'transformer'." )
                sys.exit( 1 )
            thisModel           = self._createModel( modelType=modelType,**kwargs )
        else:
            thisModel           = model
        thisModel.eval(  )
        nSamples                = 0
        nCorrectSamples         = 0
        self.dataSets           = [ ]
        self.blockWeights       = [ ]
        allYPred                = [  ]
        allYTrue                = [  ]
        self.loadData( pathTest,blockWeights )
        for k,testDS in enumerate(  self.dataSets  ):
            X_test,y_test,device    = self.extractFeatures( testDS,blockWeight=blockWeights[ k ] )
            for i in range( X_test.shape[0]//batchSize ):
                small_X_test        = X_test[  i*batchSize:i*batchSize+batchSize  ]
                small_X_test        = small_X_test.to( device )
                with torch.no_grad(  ):
                    yPred               = thisModel( small_X_test )
                    small_y_test        = y_test[  i*batchSize:i*batchSize+batchSize  ]
                    small_y_test        = small_y_test.to( device )
                    _, yPredLabels      = torch.max( yPred, dim=1 )
                    yPred_np            = yPredLabels.cpu(  ).numpy(  )
                    yTrue_np            = small_y_test.cpu(  ).numpy(  )
                    if len( yTrue_np.shape ) > 1 and yTrue_np.shape[1] > 1:
                        yTrue_np        = np.argmax( yTrue_np, axis=1 )
                    allYPred.extend( yPred_np )
                    allYTrue.extend( yTrue_np )
                    nCorrectSamples     += (yPred_np == yTrue_np).sum(  )
                    nSamples            += batchSize
        accuracy = nCorrectSamples / nSamples
        self.logger.infoMsg(f'Accuracy: {accuracy:.4f}')
        conf_matrix = confusion_matrix(  allYTrue, allYPred, labels=np.arange( 9 )  )
        classNames = [  self.nom.DICT_RIG_STATES[ i ] for i in self.nom.GOAL_RIG_STATES  ] 
        self._plotFancyContingencyTable( conf_matrix, classNames ) 
        
        