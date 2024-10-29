"""
Time series pre-annotator for the creation of well/rig activity identification engines.
Abraham C. Montes  - RAPID, UT Austin.
"""

class PreAnnotator:

    def annotate( self,dataFrame,labelCol ):
        """
        Pre-annotates a dataframe with a custom logic. 
        inputs:
        - dataFrame:        A Pandas dataframe containing the time series data.
        - labelCol:         The name of the column where annotations will be placed.
        output:             A Pandas dataframe with the labels column pre-filled in.
        """
        inProcessData       = dataFrame.copy( deep=True )
        #------------------------------------------------------------------------------------------------
        #   Implement a custom logic to pre-annotate data. The logic below corresponds to an forward chaining-based inference system for drilling operations.
        #------------------------------------------------------------------------------------------------
        from rig_state_service.RigStateIdentifier import RigStateIdentifier
        from rig_state_service.SignalProcessingControllers import InferenceEngine
        from rig_state_service.SignalProcessingControllers import StateSignalsListener
        from pathlib import Path
        rootDir                             =   Path( ).resolve( )
        iEngine                             =   InferenceEngine( str( rootDir )+r'\rig_state_service\config\config.json' )
        listener                            =   StateSignalsListener( str( rootDir )+r'\rig_state_service\config\config.json',None )
        identifier                          =   RigStateIdentifier( inferenceEngine=iEngine,signalsListener=listener )
        states                              =   [ ]
        for row in inProcessData.iterrows(  ):
            stateNo,_,_                     =       identifier.run( **dict( row[ 1 ] ) )
            states.append( stateNo )
        inProcessData[ labelCol ]           =   states
        print(f"Finished rig state update. A total of {len(states)} states were added to the DF, which in turn, has {inProcessData.shape[0]} rows")
        return inProcessData
    