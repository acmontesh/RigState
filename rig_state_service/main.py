# Copyright 2024 RAPID Consortium, The University of Texas at Austin, all rights reserved.
# Author: Abraham C. Montes
# ----------------------------------------------------------------------------------------
# Unauthorized reproduction, manipulation, distribution, or
# use of this code, in whole or in part, is strictly prohibited
# without the express written permission of The University of Texas.
# Contact: abraham.montes@utexas.edu
# ----------------------------------------------------------------------------------------

from RigStateIdentifier import RigStateIdentifier
from SignalProcessingControllers import InferenceEngine
from SignalProcessingControllers import StateSignalsListener
from kafka import KafkaConsumer, KafkaProducer
from flask import Flask
from flask_cors import CORS
import json
import time
from threading import Thread

#Creating the API and enabling cross-origin resource sharing from all sources
app                                 =   Flask( __name__ )
CORS(  app  )

#Creating instances of the inference engine and the rig state identifier
iEngine                             =   InferenceEngine( './config/config.json' )
listener                            =   StateSignalsListener( './config/config.json' )
identifier                          =   RigStateIdentifier( inferenceEngine=iEngine,signalsListener=listener )

#Connection to the broker
def connectToBroker(   ):
    while True:
        try:
            print( 'Trying to subscribe and consume topic...',flush=True )
            consumer                                = KafkaConsumer( bootstrap_servers="kafka:9092",group_id="back-consumers-rawsignals" ) 
            producer                                = KafkaProducer( bootstrap_servers="kafka:9092",
                                                        value_serializer=lambda x: json.dumps( x ).encode( 'utf-8' ),
                                                        retries=0 )
            print( 'Producer Object: {0}'.format( producer ),flush=True )
            return consumer,producer
        except Exception as e:
            print( "Unable to connect to Kafka broker... Retrying in 5 seconds" )
            time.sleep( 5 )

#Creation of the consumer
def consumeBroker(   ):
    consumer,producer                               = connectToBroker(  )
    consumer.subscribe( 'rawrigdata' )
    for messages in consumer:
        try:
            msg                                     =       json.loads(  messages.value  )
            state,stateDescription,signalsDict      =       identifier.run( **msg )
            if state:                   
                producer.send( "rigstatus",key=b"state",value={ "state":state,"description":stateDescription } )
                producer.send( "filteredrigdata",key=b"data",value={ "data":signalsDict } )
                producer.send( "jointrigdata",key=b"joint",value={ "state":stateDescription, "data":signalsDict } )
                producer.flush(  )
            continue
        except Exception as e:
            print(  "Houston, there was an error: {0}".format( e ), flush=True  )
            time.sleep( 1 )
            continue    

#API route to obtain the list of states
@app.route("/listofstates")
def getListOfStates(  ):
    arrayStates                                     = iEngine.getGoalsDescriptions(  )
    reObject                                        = { "data":arrayStates }
    return json.dumps( reObject )

if __name__ == '__main__':    
    kafkaThread                                     = Thread(  target=consumeBroker  )
    kafkaThread.start(  )
    app.run( host='0.0.0.0', port=8080, debug=True )