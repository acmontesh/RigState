# Copyright 2024 RAPID Consortium, The University of Texas at Austin, all rights reserved.
# Author: Abraham C. Montes
# ----------------------------------------------------------------------------------------
# Unauthorized reproduction, manipulation, distribution, or
# use of this code, in whole or in part, is strictly prohibited
# without the express written permission of The University of Texas.
# Contact: abraham.montes@utexas.edu
# ----------------------------------------------------------------------------------------

class BrokerIntermediary:
    templateChannelNames                            = {
        "ROP":                                        ["ROPA","Rate of Penetration (feet_h)","Rate of Penetration (ft/h)","ROP [ft/h]","Rate Of Penetration (ft_per_hr)"],
        "Flow Rate":                                  ["GPM","MFIA","Pump Output (gpm)","Pump Output [gpm]","Total Pump Output (gal_per_min)"],
        "Rotation":                                   ["RPMA","Tod Drive Rotary (RPM)","Rotary Speed","Rotary RPM (RPM)"],
        "Block Position":                             ["BPOS","Block Position (feet)", "Block Position [ft]","Block Height (feet)"],
        "Hook Load":                                  ["HKLA","Hook Load (klbs)","Hook Load [klb]"]
    }
    nRetriesConnectionToBroker                      = 3
    timeBetweenRetriesConnectionToBroker            = 5 #Seconds
    topicName                                       = 'rigstatus'

    def __init__(  self,brokerHost:str  ):
        self.producer,self.consumer                 =   self._initializeProducerAndConsumer( brokerHost )

    def extendTemplateChannelNames( self,**kwargs ):
        for key in kwargs:
            if key in list( self.templateChannelNames.keys( ) ):
                self.templateChannelNames[ key ].extend(  kwargs[ key ]  )
            else:
                self.templateChannelNames[ key ]    =   kwargs[ key ]
    
    def setRetriesToConnect( self,**kwargs ):
        if 'time' in kwargs:                        self.timeBetweenRetriesConnectionToBroker=kwargs[ 'time' ]
        if 'attempts' in kwargs:                    self.nRetriesConnectionToBroker=kwargs[ 'attempts' ]
    
    def setTopicName( self,topic ):
        self.topicName                              =   topic

    def _initializeProducerAndConsumer( self,address ):
        from kafka import KafkaProducer, KafkaConsumer
        import time
        import json
        nRetries                                    =   0
        while nRetries<self.nRetriesConnectionToBroker:
            try:
                self.producer                       =   KafkaProducer( bootstrap_servers=address,
                                                                       value_serializer=lambda x: json.dumps( x ).encode( 'utf-8' ) )
                self.consumer                       =   KafkaConsumer( bootstrap_servers=address ) 
                print( "Proxy successfully connected to the broker at {0}".format( address ),flush=True )
                return self.producer,self.consumer
            except Exception as e:
                print( "Error while connecting to the broker. Retrying in {0} seconds".format( self.timeBetweenRetriesConnectionToBroker ),flush=True )
                nRetries                            +=1
                time.sleep( self.timeBetweenRetriesConnectionToBroker )
        print( "Unable to connect to the broker after {0} attempts. Exiting".format( self.nRetriesConnectionToBroker ),flush=True )
        return None

    def broadcastData( self,pathDataFile ):
        import json        
        print( "Loading data to memory", flush=True )
        dataFrame                                   = self._loadDataToMemory( pathDataFile )
        print( "Data successfully stored in memory", flush=True )
        newCols,oldCols                             = self._identifyColumnsToExtract( dataFrame )
        dataFrame                                   = dataFrame[ list( oldCols ) ]
        dataFrame                                   = self._renameColumns( dataFrame,list( newCols ),list( oldCols ) )
        for idx,row in dataFrame.iterrows(  ):
            rowDict                                 = dict( row )
            print( rowDict,flush=True )
            self.sendDataPoint( key="key",valueDict=rowDict )

    def sendDataPoint( self,key,valueDict ):
        import time
        try:
            self.producer.send( self.topicName,key=key.encode( 'utf-8' ),value=valueDict )
            print( "Produced a data point successfully", flush=True )
            time.sleep( 2 )
        except Exception as e:
            print( "Error while producing a row of the dataframe to the Kafka broker: {0}".format( e ), flush=True )
            time.sleep( 2 )
        self.producer.flush(  )

    def _identifyColumnsToExtract( self,dataFrame ):
        columnsDF                                   = list( dataFrame.columns )
        columnsTuples                               = [ (key,col) for key in self.templateChannelNames for col in columnsDF if col in self.templateChannelNames[ key ]  ]
        newCols,oldCols                             = zip( *columnsTuples )
        return newCols,oldCols

    def _renameColumns( self,dataFrame,newCols,oldCols ):
        renameDict                                  = dict( zip( *[oldCols, newCols] ) )
        dataFrame                                   = dataFrame.rename( renameDict,axis=1 )
        return dataFrame

    def _loadDataToMemory( self,pathDataFile ):
        import pandas as pd
        return pd.read_csv( pathDataFile )