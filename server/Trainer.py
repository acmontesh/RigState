import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import scipy as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
import joblib
from Logger import LoggerDev
from Nomenclature import Nomenclature
from Models import TimeSeriesTransformer, LSTMClassifier

class Trainer:

    FDICT_PLOTS                 = {'family':'Arial','size':8}
    LOWER_LIM_HKL_FOR_BLOCK_WEIGHT      = 200
    UPPER_LIM_HKL_FOR_BLOCK_WEIGHT      = 1000

    def __init__( self ):
        self.logger             = LoggerDev(  )
        self.dataSets           = [  ]
        self.concatDataset      = [  ]
        self.blockWeights       = [  ]
        self.nom                = Nomenclature(  )
        self.labelPreds         = {  }
        self.scaler             = StandardScaler( )
        self.slidingWindow      = None

    def setSlidingWindow( self,slidingWindow ):
        self.slidingWindow      = slidingWindow

    def loadData( self,pathFolder,blockWeights={  },nBinsBWInference=100,dropNaNs=True ):
        for file in os.listdir( pathFolder ):
            if file==".ipynb_checkpoints":  continue
            df                                  = pd.read_csv( pathFolder+"/"+file,parse_dates=[ self.nom.DATE_MNEMONIC ],na_values=[-999.25] )
            if dropNaNs:                        df                                  = df.dropna( axis=0 )
            if file[ :-4 ] in list( blockWeights.keys(  ) ):
                bW                              =   blockWeights[ file[ :-4 ] ]
            else:
                bW                              =   self._getBlockWeight( df,nbins=nBinsBWInference )
                self.logger.warningMsg( f"The block weight for the {file[ :-4 ]} well was not provided. Therefore, it has been inferred. Its value is: {bW}" )
            df[ self.nom.BLOCK_WEIGHT_MNEMO ]   = bW
            self.dataSets.append( df )
            self.blockWeights.append( bW )
            self.logger.infoMsg( f"Block weight for {file} has been set on: {bW} klb." )
        self.logger.infoMsg( f"Data has been loaded correctly to the trainer. In total, {len( self.dataSets )} dataframes have been loaded." )

    def _loadCheckPoint( self, currentCheckPointPath, model, optimizer ):
        checkpoint                              = torch.load(  currentCheckPointPath, weights_only=False  )
        model.load_state_dict(  checkpoint['model_state_dict']  )
        optimizer.load_state_dict(  checkpoint['optimizer_state_dict']  )
        return model, optimizer

    def trainModel(  self, modelType, batchSize=32, nEpochs=200, learningRate=0.0001, currentCheckPointPath=None,
                    saveModel=True, savePath="model_transformer.pth", saveScaler=True, scalerPath="scaler_transformer.pkl", **kwargs  ):
        if len( self.dataSets )==0:
            self.logger.errorMsg( "No data has been loaded to the trainer. Use the loadData( ) function before training a model." )
            sys.exit( 1 );
        model, optimizer        = self._createModel( modelType,learningRate, **kwargs )
        self.logger.infoMsg( f"The {modelType} model has been created." )
        currEpoch               = 0
        trainLosses             = np.zeros( nEpochs*len( self.dataSets ) )
        if 'slidingWindow' not in kwargs:
            self.slidingWindow  = 5
            self.logger.warningMsg( f"The size of the sliding window has not been specified. By default, it has been set on {self.slidingWindow}." )
        else: self.slidingWindow= kwargs['slidingWindow']
        if currentCheckPointPath is not None:
            model, optimizer    = self._loadCheckPoint( currentCheckPointPath, model, optimizer )
            self.logger.infoMsg( "Checkpoint loaded successfully. Resuming training..." )
        else:                   self.logger.infoMsg( "Starting training..." )
        device                  = torch.device( "cuda:0" if torch.cuda.is_available( ) else "cpu" )
        typeDevice              = "GPU" if torch.cuda.is_available( ) else "CPU"
        self.logger.infoMsg(  f"Working on: {typeDevice}, model {torch.cuda.get_device_name(0)}"  )
        self.logger.infoMsg( "Extracting features for dataset No. 1." )
        X_train, y_train        = self.extractFeatures( self.dataSets[ 0 ],self.blockWeights[ 0 ],modelType=modelType,slidingWindow=self.slidingWindow )
        if len( self.dataSets )>1:
            for k,ds in enumerate( self.dataSets[1:] ):
                self.logger.infoMsg( f"Extracting features for dataset No. {k+2}." )
                X_train_temp, y_train_temp  = self.extractFeatures( ds,self.blockWeights[ k+1 ],modelType=modelType,slidingWindow=self.slidingWindow )
                X_train         = np.concatenate( [X_train,X_train_temp],axis=0 )
                y_train         = np.concatenate( [y_train,y_train_temp] )
        self.logger.infoMsg( f"The training dataset has the following shape: {X_train.shape[0]} X {X_train.shape[1]}. The response: {y_train.shape}" )
        X_train                 = self.scaler.transform( X_train )
        X_train, y_train        = self.wrapTensor( X_train,y_train,slidingWindow=self.slidingWindow )
        nNans                   = np.isnan( X_train ).sum(  )
        if nNans>0:             self.logger.warningMsg( f"There are {nNans} NaNs in the training dataset. Consider trimming NaNs before training." )
        X_train                 = torch.from_numpy( X_train.astype(np.float32) ).to( device )
        y_train                 = torch.from_numpy( y_train.astype(np.float32) )
        y_train                 = torch.argmax( y_train, dim=1 ).to( device )
        self.logger.infoMsg( f"Size of the training data: {(X_train.numel(  ) * X_train.element_size(  ))/1E9:.2f} GB for the input matrix and {(y_train.numel(  ) * y_train.element_size(  ))/1E9:.2f} GB for the response variable." )
        if saveScaler:
            joblib.dump( self.scaler, scalerPath )
            self.logger.infoMsg( f"Successfully saved scaler: {scalerPath}" )
        criterion               = nn.CrossEntropyLoss(  )
        dataset                 = TensorDataset( X_train, y_train )
        dataLoader              = DataLoader( dataset, batch_size=batchSize, shuffle=True )
        self.logger.infoMsg( "Data loaded to the torch DataLoader object." )
        for epoch in range( nEpochs ):
            if epoch==0: self.logger.infoMsg( f"Initiating forward pass (epoch {epoch}) for dataset {k}." )
            model.train(  )
            runningLoss         = 0.0
            correctPreds        = 0
            totalPreds          = 0
            for batchNo, ( inputs, labels ) in enumerate( dataLoader ):
                if batchNo%150==0:self.logger.infoMsg( f'[LOADER>>>]..... Processing batch No. {batchNo}.')
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
            self.logger.infoMsg( f'[TRAINING MSG>>>]..... Epoch {epoch+1}/{nEpochs}, Train Loss: {loss.item(  ):.4f}')
            self._saveCheckpoint( model,optimizer,epoch,trainLosses[-1],modelType,codeName=f"_{k}" )
            self.logger.infoMsg( f"Successfully saved checkpoint: {modelType}_{k}.cpt" )
        if saveModel:
            torch.save(  model.state_dict(  ), savePath  )
            self.logger.infoMsg( f"Successfully saved {modelType} model: {savePath}" )
        np.save( "training_losses.npy",trainLosses )
        return trainLosses, model
    
    def wrapTensor( self,X,y,slidingWindow ):
        XF                      = [ ]
        yF                      = [ ]
        for t in range( X.shape[ 0 ]-slidingWindow ):
            x                   = X[ t:t+slidingWindow,: ]
            XF.append( x )
            ny                  = y[ t+slidingWindow-1,: ]
            yF.append( ny )
        XF                      = np.array( XF ).reshape( -1,slidingWindow,X.shape[1] )
        yF                      = np.array( yF ).reshape( -1,y.shape[1] )
        # if modelType=="lstm":
        #     y_train                 = torch.from_numpy(yF.astype(np.float32)).to( device )
        # else:
        #     y_train                 = torch.from_numpy(yF.astype(np.float32))
        #     y_train                 = torch.argmax( y_train, dim=1 ).to( device )
        return XF.astype(np.float32),yF.astype(np.float32)

    def _getBlockWeight( self,dataSet,nbins=100 ):
        df                      = dataSet.copy( deep=True )
        df['Hook Load [klb]']   = np.where( (df[ self.nom.HOOK_LOAD_MNEMO ]<0) | (df[ self.nom.HOOK_LOAD_MNEMO ]>self.UPPER_LIM_HKL_FOR_BLOCK_WEIGHT), 0, df[ self.nom.HOOK_LOAD_MNEMO ])
        b                       = np.flip( np.argsort( np.histogram( df[ self.nom.HOOK_LOAD_MNEMO ],bins=nbins )[0] ) )
        c                       = np.histogram( df[ self.nom.HOOK_LOAD_MNEMO ],bins=nbins )[1][b][  np.histogram( df[ self.nom.HOOK_LOAD_MNEMO ],bins=nbins )[1][b]<self.LOWER_LIM_HKL_FOR_BLOCK_WEIGHT  ]
        delta                   = ( np.histogram( df[ self.nom.HOOK_LOAD_MNEMO ],bins=nbins )[1].max(  ) - np.histogram( df[ self.nom.HOOK_LOAD_MNEMO ],bins=nbins )[1].min(  ) )/(  2*nbins  )
        inferredBW              = c[0]+delta
        return inferredBW

    def extractFeatures( self, df, blockWeight, modelType,slidingWindow=5 ):
        df[self.nom.EFF_HOOK_LOAD_MNEMO]            = np.where(df[self.nom.HOOK_LOAD_MNEMO]>0,df[self.nom.HOOK_LOAD_MNEMO]-df[self.nom.BLOCK_WEIGHT_MNEMO],df[self.nom.HOOK_LOAD_MNEMO])
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
        self.scaler.partial_fit( X )
        return X, y

    def plotLosses( self, trainingLoss, testLoss, outPath="loss_curve.png" ):
        fig,ax                    = plt.subplots( figsize=(6.83,3.33) )
        ax.plot( np.arange( len(trainingLoss) ), trainingLoss, lw=1.5, color='teal',label="Training Loss" )
        if not testLoss is None: ax.plot( np.arange( len(testLoss) ), testLoss, lw=1.5, color='red',ls='--',label="Test Loss" )
        ax.set_xlabel( 'Epoch', fontdict={ **self.FDICT_PLOTS,'weight':'bold'} )
        ax.set_ylabel( 'Loss (Cross Entropy)', fontdict={ **self.FDICT_PLOTS,'weight':'bold'} )
        ax.set_xticklabels( ax.get_xticklabels( ), fontdict=self.FDICT_PLOTS )
        ax.set_yticklabels( ax.get_yticklabels( ), fontdict=self.FDICT_PLOTS )
        ax.yaxis.set_minor_locator( AutoMinorLocator( ) )
        ax.xaxis.set_minor_locator( AutoMinorLocator( ) )
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

    def testModel( self,pathTest,pathScaler=None,blockWeights={  },batchSize=32,loadFromPath=None,modelType=None,model=None,**kwargs ):
        device                  = torch.device( "cuda:0" if torch.cuda.is_available( ) else "cpu" )
        typeDevice              = "GPU" if torch.cuda.is_available( ) else "CPU"
        self.logger.infoMsg(  f"Testing on: {typeDevice}, model {torch.cuda.get_device_name(0)}"  )
        if 'slidingWindow' not in kwargs:
            self.slidingWindow  = 5
            self.logger.warningMsg( f"The size of the sliding window has not been specified. By default, it has been set on {self.slidingWindow}." )
        else: self.slidingWindow= kwargs['slidingWindow']
        if (loadFromPath is None) & (model is None):
            self.logger.errorMsg( "Cannot test a model if no model is indicated. Either specify the torch model object or a path to a .pth file containing a model." )
            sys.exit( 1 )
        if loadFromPath is not None:
            if modelType is None:
                self.logger.errorMsg( "The type of model must be specified. Options: 'lstm', 'transformer'." )
                sys.exit( 1 )
            thisModel,optimizer = self._createModel( modelType=modelType,**kwargs )
            thisModel,optimizer    = self._loadCheckPoint( loadFromPath, thisModel, optimizer )
            self.logger.infoMsg( "Checkpoint loaded successfully. Resuming training..." )
        if pathScaler is None:
            self.logger.errorMsg( "No scaler has been specified. This is required in order to test the model. Make sure you provide the path to the scaler." )
            sys.exit( 1 )
        self.scaler             = joblib.load( pathScaler )
        self.logger.infoMsg( "The scaler has been loaded successfully." )
        thisModel.eval(  )
        nSamples                = 0
        nCorrectSamples         = 0
        self.dataSets           = [ ]
        self.blockWeights       = [ ]
        allYPred                = [  ]
        allYTrue                = [  ]
        self.loadData( pathTest,blockWeights )
        X_test, y_test        = self.extractFeatures( self.dataSets[ 0 ],self.blockWeights[ 0 ],modelType=modelType,slidingWindow=self.slidingWindow )
        if len( self.dataSets )>1:
            for k,ds in enumerate( self.dataSets[1:] ):
                self.logger.infoMsg( f"Extracting features for dataset No. {k+2}." )
                X_test_temp, y_test_temp    = self.extractFeatures( ds,self.blockWeights[ k+1 ],modelType=modelType,slidingWindow=self.slidingWindow )
                X_test                      = np.concatenate( [X_test,X_test_temp],axis=0 )
                y_test                      = np.concatenate( [y_test,y_test_temp] )
        self.logger.infoMsg( f"The testing dataset has the following shape: {X_test.shape[0]} X {X_test.shape[1]}. The response: {y_test.shape}." )
        X_test                  = self.scaler.transform( X_test )
        X_test, y_test          = self.wrapTensor( X_test, y_test,slidingWindow=self.slidingWindow )
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

