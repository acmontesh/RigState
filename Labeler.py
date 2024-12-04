from Logger import LoggerDev
from Nomenclature import Nomenclature
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
from Models import TimeSeriesTransformer, LSTMClassifier, ConvLSTMClassifier, ConvTimeSeriesTransformer

class RigStateIdentifier:
    def __init__( self,typeModel,pathChkPt,pathScaler,**kwargs ):
        self.type       = typeModel
        self.logger     = LoggerDev(  )
        self.nom        = Nomenclature(  )
        self.model      = self._loadModel( pathChkPt,typeModel,**kwargs )
        self._loadScaler( pathScaler )
        self.blockWeights=[  ]
        self.inQueueDataset=[  ]

    def _loadScaler( self,pathScaler=None ):
        if pathScaler is None:
            self.logger.errorMsg( "No scaler has been specified. This is required in order to test the model. Make sure you provide the path to the scaler." )
            sys.exit( 1 )
        self.scaler             = joblib.load( pathScaler )
        self.logger.infoMsg( "The scaler has been loaded successfully." )

    def loadData( self,preloadedDF=None ):
        if self.nom.BLOCK_WEIGHT_MNEMO in list( preloadedDF.columns ):
            self.inQueueDataset.append( preloadedDF )
            self.blockWeights.append( preloadedDF[self.nom.BLOCK_WEIGHT_MNEMO].iloc[0] )
            self.logger.infoMsg( f"Loaded provided dataframe sucessfully." )
        else:
            self.logger.errorMsg( f"The provided DF does not contain the block weight. Please include it before loading." )
            sys.exit( 1 )

    def wrapTensor( self,X,slidingWindow ):
        XF                      = [ ]
        for t in range( X.shape[ 0 ]-slidingWindow ):
            x                   = X[ t:t+slidingWindow,: ]
            XF.append( x )
        XF                      = np.array( XF ).reshape( -1,slidingWindow,X.shape[1] )
        return XF.astype(np.float32)

    def predict( self,pathTest=None,blockWeights={  },loadFromPath=None,modelType=None,model=None,fitScaler=False,preloadedDF=None,profilingTime=False,**kwargs ):
        device                  = torch.device( "cuda:0" if torch.cuda.is_available( ) else "cpu" )
        typeDevice              = "GPU" if torch.cuda.is_available( ) else "CPU"
        self.logger.infoMsg(  f"Predicting on: {typeDevice}, model {torch.cuda.get_device_name(0)}"  )
        if 'slidingWindow' not in kwargs:
            self.slidingWindow  = [5,30]
            self.logger.warningMsg( f"The size of the sliding windows (two resolution levels) has not been specified. By default, the small time window has been set on {self.slidingWindow[0]} sec, and the large one to {self.slidingWindow[1]} sec." )
        else: self.slidingWindow= kwargs['slidingWindow']
        if 'slidingWindowCoverage' not in kwargs:
            self.slidingWindowCoverage  = 5
            self.logger.warningMsg( f"The size of the transformation window has not been specified. By default, it has been set on {self.slidingWindowCoverage}." )
        else: self.slidingWindowCoverage= kwargs['slidingWindowCoverage']           
        self.model.eval(  )
        self.loadData( preloadedDF=preloadedDF )
        XPred                   = self.extractFeatures( self.inQueueDataset[ 0 ],slidingWindow=self.slidingWindow,fitScaler=fitScaler )
        XPred                   = self.scaler.transform( XPred )
        XPred                   = self.wrapTensor( XPred,slidingWindow=self.slidingWindowCoverage )
        self.logger.infoMsg( f"After wrapping, the shape of X is: {XPred.shape}" )
        XPred                   = torch.from_numpy( XPred.astype(np.float32) )
        XPred                   = XPred.to( device )
        with torch.no_grad(  ):
            yPred               = self.model( XPred )
            _, yPredLabels      = torch.max( yPred, dim=1 )
        return yPredLabels.cpu(  ).numpy(  )

    def _loadModel( self,loadFromPath,modelType,**kwargs ):
        if loadFromPath is not None:
            if modelType is None:
                self.logger.errorMsg( "The type of model must be specified. Options: 'lstm', 'transformer'." )
                sys.exit( 1 )
            thisModel,optimizer = self._createModel( modelType=modelType,**kwargs )
            thisModel,optimizer    = self._loadCheckPoint( loadFromPath, thisModel, optimizer )
            self.logger.infoMsg( "Checkpoint loaded successfully. Resuming training..." )
        return thisModel

    def _createModel( self,modelType,learningRate=0.0001,**kwargs ):
        device = torch.device( "cuda:0" if torch.cuda.is_available( ) else "cpu" )
        if modelType==self.nom.TRANSFORMER_MODEL_MNEMO:
            model                   = self.createModelTransformer( device, **kwargs )
        elif modelType==self.nom.LSTM_MODEL_MNEMO:
            model                   = self.createModelLSTM( device, **kwargs )
        elif modelType==self.nom.CONV_LSTM_MODEL_MNEMO:
            model                   = self.createModelConvLSTM( device, **kwargs )
        elif modelType==self.nom.CONV_TRANSFORMER_MODEL_MNEMO:
            model                   = self.createModelConvTransformer( device, **kwargs )
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
    
    def createModelConvTransformer( self, device, **kwargs ):
        nInputs         = 10 if "nInputs" not in kwargs else kwargs["nInputs"]
        nLayers         = 3 if "nLayers" not in kwargs else kwargs["nLayers"]
        dModel          = 256 if "dModel" not in kwargs else kwargs["dModel"]
        nOutput         = 11 if "nOutput" not in kwargs else kwargs["nOutput"]
        nHead           = 8 if "nHead" not in kwargs else kwargs["nHead"]
        dimFeedForward  = 512 if "dimFeedForward" not in kwargs else kwargs["dimFeedForward"]
        model           = ConvTimeSeriesTransformer(  nInputs=nInputs,nLayers=nLayers,dModel=dModel,
                                        nOutput=nOutput,nHead=nHead,dimFeedForward=dimFeedForward )
        model.to( device )
        return model

    def createModelLSTM( self,device, **kwargs ):
        nInputs         = 10 if "nInputs" not in kwargs else kwargs["nInputs"]
        nLayers         = 3 if "nLayers" not in kwargs else kwargs["nLayers"]
        nHidden         = 30 if "nHidden" not in kwargs else kwargs["nHidden"]
        nOutput         = 11 if "nOutput" not in kwargs else kwargs["nOutput"]
        model           = LSTMClassifier(  nInputs=nInputs,nLayers=nLayers, nOutput=nOutput,nHidden=nHidden )
        model.to( device )
        return model
    
    def createModelConvLSTM( self,device, **kwargs ):
        nInputs         = 10 if "nInputs" not in kwargs else kwargs["nInputs"]
        nLayers         = 3 if "nLayers" not in kwargs else kwargs["nLayers"]
        nHidden         = 30 if "nHidden" not in kwargs else kwargs["nHidden"]
        nOutput         = 11 if "nOutput" not in kwargs else kwargs["nOutput"]
        model           = ConvLSTMClassifier(  nInputs=nInputs,nLayers=nLayers, nOutput=nOutput,nHidden=nHidden )
        model.to( device )
        return model

    def _trend( self,window ):
        slope,_                 = np.polyfit( np.arange(len(window)), window, deg=1 )
        return slope


    def extractFeatures( self, df, slidingWindow=[2,5,10,30],fitScaler=False ):
        if isinstance( slidingWindow,list ):
            if len( slidingWindow )>=2:
                if len( slidingWindow )==2:
                    shortTW,longTW                                          = slidingWindow
                elif len( slidingWindow )==4:
                    shortTW,intTW1,intTW2,longTW                            = slidingWindow
                else:
                    self.logger.errorMsg( "You must provide either two or four time windows for window-based features derivation." )
                    sys.exit( 1 )
            else:
                self.logger.errorMsg( "slidingWindow must be a list with at least two components: The size of the time windows for inspecting short- and long-term variation")
                sys.exit( 1 )
        else:
            self.logger.errorMsg( "slidingWindow must be a list." )
            sys.exit( 1 )
        df[self.nom.EFF_HOOK_LOAD_MNEMO]            = np.where(df[self.nom.HOOK_LOAD_MNEMO]>0,df[self.nom.HOOK_LOAD_MNEMO]-df[self.nom.BLOCK_WEIGHT_MNEMO],0)
        df[self.nom.BLOCK_POSITION_TREND_SHORT_MNEMO]= df[self.nom.BLOCK_POSITION_MNEMO].rolling(window=shortTW).apply(self._trend,raw=True,engine='cython')
        df[self.nom.BLOCK_POSITION_TREND_LONG_MNEMO]= df[self.nom.BLOCK_POSITION_MNEMO].rolling(window=longTW).apply(self._trend,raw=True,engine='cython')
        df[self.nom.HOOK_LOAD_MEAN_MNEMO]           = df[self.nom.EFF_HOOK_LOAD_MNEMO].rolling(window=shortTW).mean(  )
        df[self.nom.HOOK_LOAD_SHORT_TREND]                  = df[self.nom.EFF_HOOK_LOAD_MNEMO].rolling(window=shortTW).apply(self._trend,raw=True,engine='cython')
        df[self.nom.HOOK_LOAD_LONG_TREND]                   = df[self.nom.EFF_HOOK_LOAD_MNEMO].rolling(window=longTW).apply(self._trend,raw=True,engine='cython')
        if len( slidingWindow )==4:
            df[self.nom.BLOCK_POSITION_TREND_IT1_MNEMO]     = df[self.nom.BLOCK_POSITION_MNEMO].rolling(window=intTW1).apply(self._trend,raw=True,engine='cython')
            df[self.nom.BLOCK_POSITION_TREND_IT2_MNEMO]= df[self.nom.BLOCK_POSITION_MNEMO].rolling(window=intTW2).apply(self._trend,raw=True,engine='cython')
            df[self.nom.HOOK_LOAD_TREND_IT1_MNEMO]                = df[self.nom.EFF_HOOK_LOAD_MNEMO].rolling(window=intTW1).apply(self._trend,raw=True,engine='cython')
            df[self.nom.HOOK_LOAD_TREND_IT2_MNEMO]                = df[self.nom.EFF_HOOK_LOAD_MNEMO].rolling(window=intTW2).apply(self._trend,raw=True,engine='cython')
        df[self.nom.FLOW_RATE_VARIABILITY_MNEMO]    = df[self.nom.FLOW_IN_MNEMO].rolling(window=shortTW).std(  )
        df[self.nom.FLOW_RATE_MEAN_MNEMO]           = df[self.nom.FLOW_IN_MNEMO].rolling(window=shortTW).mean(  )
        df[self.nom.PRESSURE_MEAN_MNEMO]            = df[self.nom.STANDPIPE_PRESSURE_MNEMO].rolling(window=shortTW).mean(  )
        df[self.nom.RPM_MEAN_MNEMO]                 = df[self.nom.RPM_MNEMO].rolling(window=shortTW).mean(  )
        df[self.nom.ROP_MEAN_MNEMO]                 = df[self.nom.ROP_MNEMO].rolling(window=shortTW).mean(  )
        if len( slidingWindow )==4:
            dfTraining                                  = df[[self.nom.BLOCK_POSITION_TREND_SHORT_MNEMO,
                                                            self.nom.BLOCK_POSITION_TREND_LONG_MNEMO,
                                                            self.nom.BLOCK_POSITION_TREND_IT1_MNEMO,
                                                            self.nom.BLOCK_POSITION_TREND_IT2_MNEMO,
                                                                self.nom.FLOW_RATE_VARIABILITY_MNEMO,
                                                                self.nom.FLOW_RATE_MEAN_MNEMO,
                                                                self.nom.PRESSURE_MEAN_MNEMO,
                                                                self.nom.RPM_MEAN_MNEMO,
                                                                self.nom.HOOK_LOAD_MEAN_MNEMO,
                                                                self.nom.ROP_MEAN_MNEMO,
                                                                self.nom.HOOK_LOAD_SHORT_TREND,
                                                                self.nom.HOOK_LOAD_LONG_TREND,
                                                                self.nom.HOOK_LOAD_TREND_IT1_MNEMO,
                                                                self.nom.HOOK_LOAD_TREND_IT2_MNEMO]]
        elif len( slidingWindow )==2:
            dfTraining                                  = df[[self.nom.BLOCK_POSITION_TREND_SHORT_MNEMO,
                                                            self.nom.BLOCK_POSITION_TREND_LONG_MNEMO,
                                                                self.nom.FLOW_RATE_VARIABILITY_MNEMO,
                                                                self.nom.FLOW_RATE_MEAN_MNEMO,
                                                                self.nom.PRESSURE_MEAN_MNEMO,
                                                                self.nom.RPM_MEAN_MNEMO,
                                                                self.nom.HOOK_LOAD_MEAN_MNEMO,
                                                                self.nom.ROP_MEAN_MNEMO,
                                                                self.nom.HOOK_LOAD_SHORT_TREND,
                                                                self.nom.HOOK_LOAD_LONG_TREND]]
        self.logger.infoMsg( f"Before dropping NaNs during feature extraction, the number of observations in X is {dfTraining.shape[0]}" )
        dfTraining                                  = dfTraining.dropna( axis=0 )
        self.logger.infoMsg( f"After dropping NaNs during feature extraction, the number of observations in X is {dfTraining.shape[0]}" )
        X                       = dfTraining.values
        if fitScaler:           self.scaler.partial_fit( X )
        return X  
    
    def _loadCheckPoint( self, currentCheckPointPath, model, optimizer ):
        checkpoint                              = torch.load(  currentCheckPointPath, weights_only=False  )
        model.load_state_dict(  checkpoint['model_state_dict']  )
        optimizer.load_state_dict(  checkpoint['optimizer_state_dict']  )
        return model, optimizer
    