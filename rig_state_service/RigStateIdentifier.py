# Copyright 2024 RAPID Consortium, The University of Texas at Austin, all rights reserved.
# Author: Abraham C. Montes
# ----------------------------------------------------------------------------------------
# Unauthorized reproduction, manipulation, distribution, or
# use of this code, in whole or in part, is strictly prohibited
# without the express written permission of The University of Texas.
# Contact: abraham.montes@utexas.edu
# ----------------------------------------------------------------------------------------

class RigStateIdentifier:
    
    SIGNAL_NAN_PLACEHOLDER_NUMBER                               =   -999.25

    def __init__(  self, inferenceEngine, signalsListener  ):
        self.inferenceEngine		                            = inferenceEngine
        self.signalsListener                                    = signalsListener
        self.cacheSignals                                       = {  }
        self.cacheState                                         = 118

    def _updateCacheSignals( self,key, value ):
        self.cacheSignals[ key ]                                = value
    
    def _updateCacheState( self,state ):
        self.cacheState                                         = state

    def run( self,**kwargs ):
        signalsDict                                             = self._cleanNans( kwargs )
        signalsDict['Delta Time']                               = 1.0
        _,newFacts                                              = self.signalsListener.updateFactsRTSignals( signalsDict )
        self.inferenceEngine.updateFactsOnly( newFacts )
        rigState,rigStateDescription                            = self.inferenceEngine.run(  )
        if rigState:
            self._updateCacheState( rigState )
        else:
            rigState                                            = self.cacheState
            rigStateDescription                                 = 'Connection or other operations'
        return rigState,rigStateDescription,signalsDict
    
    def _cleanNans( self,kwargsDict ):
        signalsDict                                             = kwargsDict
        import math
        for key in signalsDict:
            value                                               = signalsDict[ key ]
            if isinstance( value,( int,float,complex ) ):
                if ( ( math.isnan( value ) ) | ( value==self.SIGNAL_NAN_PLACEHOLDER_NUMBER ) ):
                    if key in list( self.cacheSignals.keys(  ) ):
                        signalsDict[ key ]                          = self.cacheSignals[ key ]
                    else: 
                        signalsDict[ key ]                          = 0
                else:
                    self._updateCacheSignals( key, value )
        return signalsDict