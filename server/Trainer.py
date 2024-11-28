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
from Nomenclature import Nomenclature
from Logger import LoggerDev
from Models import TimeSeriesTransformer

class Trainer:

    FDICT_PLOTS                 = {'family':'Arial','size':8}
    LOWER_LIM_HKL_FOR_BLOCK_WEIGHT      = 100
    UPPER_LIM_HKL_FOR_BLOCK_WEIGHT      = 1000
    WINDOW_BLOCK_POS_TREND              = 3

    def __init__( self ):
        """
        slidingWindow:          Time windows to derive features. For example, the trend of the block position. 
                                It is a list. The first component is a small time window, to emphasize short-term changes. 
                                The second one is a large time window, to account for long-term variations.
        slidingWindowCoverage:  Time window of size T used to convert the extracted features from a NxD matrix to a NxTxD, where T is the size of the subsequences
                                that the model will use as inputs.
        """
        self.logger             = LoggerDev(  )
        self.dataSets           = [  ]
        self.concatDataset      = [  ]
        self.blockWeights       = [  ]
        self.nom                = Nomenclature(  )
        self.labelPreds         = {  }
        self.scaler             = StandardScaler( )
        self.slidingWindow= None
        self.slidingWindowCoverage= None

    def setSlidingWindow( self,slidingWindows ):
        self.slidingWindow      = slidingWindows

    def setSlidingWindowCoverage( self,slidingWindowCoverage ):
        self.slidingWindowCoverage      = slidingWindowCoverage

    def loadData( self,pathFolder,blockWeights={  },nBinsBWInference=100,dropNaNs=True ):
        for file in os.listdir( pathFolder ):
            if file==".ipynb_checkpoints":  continue
            df                                  = pd.read_csv( pathFolder+"/"+file,parse_dates=[ self.nom.DATE_MNEMONIC ],na_values=[-999.25] )
            if dropNaNs:                        df                                  = df.dropna( axis=0 )
            if file[ :-4 ] in list( blockWeights.keys(  ) ):
                bW                              =   blockWeights[ file[ :-4 ] ]
            else:
                bW                              =   self._getBlockWeight( df,nbins=nBinsBWInference )
                self.logger.warningMsg( f"The block weight for the {file[ :-4 ]} well was not provided. Therefore, it has been inferred. Its value is: {bW:.1f}" )
            df[ self.nom.BLOCK_WEIGHT_MNEMO ]   = bW
            self.dataSets.append( df )
            self.blockWeights.append( bW )
            self.logger.infoMsg( f"Block weight for {file} has been set on: {bW:.1f} klb." )
        self.logger.infoMsg( f"Data has been loaded correctly to the trainer. In total, {len( self.dataSets )} dataframes have been loaded." )

    def _loadCheckPoint( self, currentCheckPointPath, model, optimizer ):
        checkpoint                              = torch.load(  currentCheckPointPath, weights_only=False  )
        model.load_state_dict(  checkpoint['model_state_dict']  )
        optimizer.load_state_dict(  checkpoint['optimizer_state_dict']  )
        return model, optimizer

    def trainModel(  self, modelType, batchSize=32, nEpochs=200, learningRate=0.0001, currentCheckPointPath=None,
                    saveModel=True, savePath="model_transformer.pth", saveScaler=True, scalerPath="scaler_transformer",
                     stratifyData=True,fitScaler=True, **kwargs  ):
        if len( self.dataSets )==0:
            self.logger.errorMsg( "No data has been loaded to the trainer. Use the loadData( ) function before training a model." )
            sys.exit( 1 );
        model, optimizer        = self._createModel( modelType,learningRate, **kwargs )
        self.logger.infoMsg( f"The {modelType} model has been created." )
        currEpoch               = 0
        trainLosses             = np.zeros( nEpochs )
        decorators1          = "default_heads" if "nHeads" not in kwargs else f"H{kwargs["nHeads"]}"
        decorators2          = "default_layers" if "nLayers" not in kwargs else f"L{kwargs["nLayers"]}"
        if 'slidingWindow' not in kwargs:
            self.slidingWindow  = [5,30]
            self.logger.warningMsg( f"The size of the sliding windows (two resolution levels) has not been specified. By default, the small time window has been set on {self.slidingWindow[0]} sec, and the large one to {self.slidingWindow[1]} sec." )
        else: self.slidingWindow= kwargs['slidingWindow']
        if 'slidingWindowCoverage' not in kwargs:
            self.slidingWindowCoverage  = 5
            self.logger.warningMsg( f"The size of the transformation window has not been specified. By default, it has been set on {self.slidingWindowCoverage}." )
        else: self.slidingWindowCoverage= kwargs['slidingWindowCoverage']
        if currentCheckPointPath is not None:
            model, optimizer    = self._loadCheckPoint( currentCheckPointPath, model, optimizer )
            self.logger.infoMsg( "Checkpoint loaded successfully. Resuming training..." )
        else:                   self.logger.infoMsg( "Starting training..." )
        device                  = torch.device( "cuda:0" if torch.cuda.is_available( ) else "cpu" )
        typeDevice              = "GPU" if torch.cuda.is_available( ) else "CPU"
        self.logger.infoMsg(  f"Working on: {typeDevice}, model {torch.cuda.get_device_name(0)}"  )
        self.logger.infoMsg( "Extracting features for dataset No. 1." )
        X_train, y_train        = self.extractFeatures( self.dataSets[ 0 ],self.blockWeights[ 0 ],modelType=modelType,slidingWindow=self.slidingWindow,fitScaler=fitScaler )
        if len( self.dataSets )>1:
            for k,ds in enumerate( self.dataSets[1:] ):
                self.logger.infoMsg( f"Extracting features for dataset No. {k+2}." )
                X_train_temp, y_train_temp  = self.extractFeatures( ds,self.blockWeights[ k+1 ],modelType=modelType,slidingWindow=self.slidingWindow,fitScaler=fitScaler )
                X_train         = np.concatenate( [X_train,X_train_temp],axis=0 )
                y_train         = np.concatenate( [y_train,y_train_temp] )
        self.logger.infoMsg( f"The training dataset has the following shape: {X_train.shape[0]} X {X_train.shape[1]}. The response: {y_train.shape}" )
        X_train                 = self.scaler.transform( X_train )
        X_train, y_train        = self.wrapTensor( X_train,y_train,slidingWindow=self.slidingWindowCoverage )
        nNans                   = np.isnan( X_train ).sum(  )
        if nNans>0:             self.logger.warningMsg( f"There are {nNans} NaNs in the training dataset. Consider trimming NaNs before training." )
        X_train, y_train        = self._stratifyData( X_train, y_train,stratify=stratifyData )
        X_train                 = torch.from_numpy( X_train.astype(np.float32) ).to( device )
        y_train                 = torch.from_numpy( y_train.astype(np.float32) )
        y_train                 = torch.argmax( y_train, dim=1 ).to( device )
        self.logger.infoMsg( f"Size of the training data: {(X_train.numel(  ) * X_train.element_size(  ))/1E9:.2f} GB for the input matrix and {(y_train.numel(  ) * y_train.element_size(  ))/1E9:.2f} GB for the response variable." )
        if saveScaler:
            joblib.dump( self.scaler, f"{scalerPath}_CV{self.slidingWindowCoverage}_{decorators1}_{decorators2}.pkl" )
            self.logger.infoMsg( f"Successfully saved scaler: {scalerPath}" )
        criterion               = nn.CrossEntropyLoss(  )
        dataset                 = TensorDataset( X_train, y_train )
        dataLoader              = DataLoader( dataset, batch_size=batchSize, shuffle=True )
        self.logger.infoMsg( "Data loaded to the torch DataLoader object." )
        for epoch in range( nEpochs ):
            if epoch==0: self.logger.infoMsg( f"Initiating forward pass (epoch {epoch})." )
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
            trainLosses[ epoch ] = epochLoss
            self.logger.infoMsg( f'[TRAINING MSG>>>]..... Epoch {epoch+1}/{nEpochs}, Train Loss: {loss.item(  ):.4f}')
            self._saveCheckpoint( model,optimizer,epoch,trainLosses[-1],modelType,codeName=f"_{decorators1}_{decorators2}" )
            self.logger.infoMsg( f"Successfully saved checkpoint: {modelType}.chpt" )
        if saveModel:
            torch.save(  model.state_dict(  ), savePath  )
            self.logger.infoMsg( f"Successfully saved {modelType} model: {savePath}" )
        return trainLosses, model
    
    def _stratifyData( self,X,y,stratify,nUndersamplingRounds=2 ):
        if not stratify:    return X,y
        yComp                   = np.argmax( y,axis=1 )
        self.logger.infoMsg( f"The number of unique classes are: {len(np.unique(yComp))}" )
        countDict                   = { x:np.sum( yComp==x ) for x in np.unique( yComp ) }
        self.logger.infoMsg( f"Before undersampling, the counts per class are: {countDict}." )
        newX,newY               = ( X,y )
        newX,newY,countDict     = self._undersampling( newX,newY,nTimes=nUndersamplingRounds )
        self.logger.infoMsg( f"After undersampling, the counts per class are: {countDict}." )
        return newX,newY    
    
    def _undersampling( self,newX,newY,nTimes=2 ):
        buffX,buffY             = ( newX,newY )
        for i in range( nTimes ):
            yComp                   = np.argmax( buffY,axis=1 )
            countDict               = { x:np.sum( yComp==x ) for x in np.unique( yComp ) }
            classRef                = sorted( countDict,key=countDict.get,reverse=True )[i]
            classRefNext            = sorted( countDict,key=countDict.get,reverse=True )[i+1]
            diff                    = countDict[classRef] - countDict[classRefNext]
            rdSetDelete             = np.array( [ ] )
            for jClass in sorted( countDict,key=countDict.get,reverse=True )[0:i+1]:
                indices             = np.argwhere( yComp==jClass ).reshape( -1, )
                selection           = np.random.choice( indices, size=diff, replace=False )
                rdSetDelete         = np.concatenate( [ rdSetDelete,selection ] )
            allIdxs                 = np.setdiff1d( np.arange( buffX.shape[0] ), rdSetDelete ).astype( int )
            buffX                    = buffX[ allIdxs ]
            buffY                    = buffY[ allIdxs ]
        yComp                       = np.argmax( buffY,axis=1 )
        countDict                   = { x:np.sum( yComp==x ) for x in np.unique( yComp ) }
        return buffX,buffY,countDict

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
        df[self.nom.HOOK_LOAD_MNEMO]   = np.where( (df[ self.nom.HOOK_LOAD_MNEMO ]<0) | (df[ self.nom.HOOK_LOAD_MNEMO ]>self.UPPER_LIM_HKL_FOR_BLOCK_WEIGHT), 0, df[ self.nom.HOOK_LOAD_MNEMO ])
        b                       = np.flip( np.argsort( np.histogram( df[ self.nom.HOOK_LOAD_MNEMO ],bins=nbins )[0] ) )
        c                       = np.histogram( df[ self.nom.HOOK_LOAD_MNEMO ],bins=nbins )[1][b][  np.histogram( df[ self.nom.HOOK_LOAD_MNEMO ],bins=nbins )[1][b]<self.LOWER_LIM_HKL_FOR_BLOCK_WEIGHT  ]
        delta                   = ( np.histogram( df[ self.nom.HOOK_LOAD_MNEMO ],bins=nbins )[1].max(  ) - np.histogram( df[ self.nom.HOOK_LOAD_MNEMO ],bins=nbins )[1].min(  ) )/(  2*nbins  )
        inferredBW              = c[0]+delta
        return inferredBW

    def extractFeatures( self, df, blockWeight, modelType,slidingWindow=[5,60],fitScaler=False ):
        if isinstance( slidingWindow,list ):
            if len( slidingWindow )>=2:
                shortTW,longTW                              = slidingWindow
            else: 
                self.logger.errorMsg( "slidingWindow must be a list with at least two components: The size of the time windows for inspecting short- and long-term variations." )
                sys.exit( 1 )
        else:
            self.logger.errorMsg( "slidingWindow must be a list." )
            sys.exit( 1 )
        df[self.nom.EFF_HOOK_LOAD_MNEMO]            = np.where(df[self.nom.HOOK_LOAD_MNEMO]>0,df[self.nom.HOOK_LOAD_MNEMO]-df[self.nom.BLOCK_WEIGHT_MNEMO],0)
        df[self.nom.BLOCK_POSITION_TREND_SHORT_MNEMO]= df[self.nom.BLOCK_POSITION_MNEMO].rolling(window=shortTW).apply(self._trend,raw=True,engine='cython')
        df[self.nom.BLOCK_POSITION_TREND_LONG_MNEMO]= df[self.nom.BLOCK_POSITION_MNEMO].rolling(window=longTW).apply(self._trend,raw=True,engine='cython')
        df[self.nom.FLOW_RATE_VARIABILITY_MNEMO]    = df[self.nom.FLOW_IN_MNEMO].rolling(window=shortTW).std(  )
        df[self.nom.FLOW_RATE_MEAN_MNEMO]           = df[self.nom.FLOW_IN_MNEMO].rolling(window=shortTW).mean(  )
        df[self.nom.PRESSURE_MEAN_MNEMO]            = df[self.nom.STANDPIPE_PRESSURE_MNEMO].rolling(window=shortTW).mean(  )
        df[self.nom.RPM_MEAN_MNEMO]                 = df[self.nom.RPM_MNEMO].rolling(window=shortTW).mean(  )
        df[self.nom.HOOK_LOAD_MEAN_MNEMO]           = df[self.nom.EFF_HOOK_LOAD_MNEMO].rolling(window=shortTW).mean(  )
        df[self.nom.HOOK_LOAD_SHORT_TREND]                = df[self.nom.EFF_HOOK_LOAD_MNEMO].rolling(window=shortTW).apply(self._trend,raw=True,engine='cython')
        df[self.nom.HOOK_LOAD_LONG_TREND]                = df[self.nom.EFF_HOOK_LOAD_MNEMO].rolling(window=longTW).apply(self._trend,raw=True,engine='cython')
        # df[self.nom.HOOK_LOAD_VARIABILITY_MNEMO]    = df[self.nom.EFF_HOOK_LOAD_MNEMO].rolling(window=shortTW).std(  )
        df[self.nom.ROP_MEAN_MNEMO]                 = df[self.nom.ROP_MNEMO].rolling(window=shortTW).mean(  )
        dfTraining                                  = df[[self.nom.BLOCK_POSITION_TREND_SHORT_MNEMO,
                                                          self.nom.BLOCK_POSITION_TREND_LONG_MNEMO,
                                                            self.nom.FLOW_RATE_VARIABILITY_MNEMO,
                                                            self.nom.FLOW_RATE_MEAN_MNEMO,
                                                            self.nom.PRESSURE_MEAN_MNEMO,
                                                            self.nom.RPM_MEAN_MNEMO,
                                                            self.nom.HOOK_LOAD_MEAN_MNEMO,
                                                            self.nom.ROP_MEAN_MNEMO,
                                                            self.nom.HOOK_LOAD_SHORT_TREND,
                                                            self.nom.HOOK_LOAD_LONG_TREND,
                                                            self.nom.RIG_STATE_MNEMO]]
        dfTraining                                  = dfTraining.dropna( axis=0 )
        X                       = dfTraining.iloc[:,:-1].values
        y                       = dfTraining[self.nom.RIG_STATE_MNEMO].values
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
        newPath    = f"{modelType}{codeName}_CV{self.slidingWindowCoverage}"
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
        nInputs         = 10 if "nInputs" not in kwargs else kwargs["nInputs"]
        nLayers         = 3 if "nLayers" not in kwargs else kwargs["nLayers"]
        dModel          = 256 if "dModel" not in kwargs else kwargs["dModel"]
        nOutput         = 11 if "nOutput" not in kwargs else kwargs["nOutput"]
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

    def testModel( self,pathTest,pathScaler=None,blockWeights={  },batchSize=32,loadFromPath=None,modelType=None,model=None,fitScaler=False,**kwargs ):
        device                  = torch.device( "cuda:0" if torch.cuda.is_available( ) else "cpu" )
        typeDevice              = "GPU" if torch.cuda.is_available( ) else "CPU"
        self.logger.infoMsg(  f"Testing on: {typeDevice}, model {torch.cuda.get_device_name(0)}"  )
        if 'slidingWindow' not in kwargs:
            self.slidingWindow  = [5,30]
            self.logger.warningMsg( f"The size of the sliding windows (two resolution levels) has not been specified. By default, the small time window has been set on {self.slidingWindow[0]} sec, and the large one to {self.slidingWindow[1]} sec." )
        else: self.slidingWindow= kwargs['slidingWindow']
        if 'slidingWindowCoverage' not in kwargs:
            self.slidingWindowCoverage  = 5
            self.logger.warningMsg( f"The size of the transformation window has not been specified. By default, it has been set on {self.slidingWindowCoverage}." )
        else: self.slidingWindowCoverage= kwargs['slidingWindowCoverage']
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
        X_test, y_test        = self.extractFeatures( self.dataSets[ 0 ],self.blockWeights[ 0 ],modelType=modelType,slidingWindow=self.slidingWindow,fitScaler=fitScaler )
        if len( self.dataSets )>1:
            for k,ds in enumerate( self.dataSets[1:] ):
                self.logger.infoMsg( f"Extracting features for dataset No. {k+2}." )
                X_test_temp, y_test_temp    = self.extractFeatures( ds,self.blockWeights[ k+1 ],modelType=modelType,slidingWindow=self.slidingWindow,fitScaler=fitScaler )
                X_test                      = np.concatenate( [X_test,X_test_temp],axis=0 )
                y_test                      = np.concatenate( [y_test,y_test_temp] )
        self.logger.infoMsg( f"The testing dataset has the following shape: {X_test.shape[0]} X {X_test.shape[1]}. The response: {y_test.shape}." )
        X_test                  = self.scaler.transform( X_test )
        X_test, y_test          = self.wrapTensor( X_test, y_test,slidingWindow=self.slidingWindowCoverage )
        X_test                  = torch.from_numpy( X_test.astype(np.float32) )
        y_test                  = torch.from_numpy( y_test.astype(np.float32) )
        y_test                  = torch.argmax( y_test, dim=1 )
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
        confMatrix = confusion_matrix(  allYTrue, allYPred, labels=np.arange( 11 )  )
        classNames = [  self.nom.DICT_RIG_STATES[ i ] for i in self.nom.GOAL_RIG_STATES  ]
        self._plotFancyContingencyTable( confMatrix, classNames )
        self.printOutF1Scores( confMatrix,None,classNames )
        return np.array( allYPred )
    
    def printOutF1Scores( self,confMatrix,nClasses=None,classesNames=None ):
        if nClasses is None:
            nClasses                = confMatrix.shape[0]
            self.logger.warningMsg( f"The number of classes has been inferred from the confusion matrix: {confMatrix.shape[0]}" )
        precScores      = [  ]
        recallScores    = [  ]
        f1Scores        = [  ]
        for i in range( nClasses ):
            TP = confMatrix[i, i]
            FP = confMatrix[:, i].sum( ) - TP
            FN = confMatrix[i, :].sum( ) - TP
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = 2 * ( precision * recall ) / ( precision + recall ) if precision + recall > 0 else 0
            precScores.append(precision)
            recallScores.append(recall)
            f1Scores.append(f1)
        if classesNames is None:        classesNames = [ f"{i}" for i in range(len(f1Scores)) ]
        for i, (f1, precision, recall) in enumerate(zip( f1Scores, precScores, recallScores )):
            self.logger.resultMessage( f"Class {classesNames[i]}: Precision={precision:.2f}, Recall={recall:.2f}, F1 Score={f1:.2f}" )
        return precScores,recallScores,f1Scores

