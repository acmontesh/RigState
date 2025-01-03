from Trainer import Trainer

batch       = [
#    dict(slidingWindow=[2,30],slidingWindowCoverage=30,nHidden=50,nLayers=3),
    dict(slidingWindow=[5,30],slidingWindowCoverage=60,nHeads=8,nLayers=6),
 #   dict(slidingWindow=[2,30],slidingWindowCoverage=15,nHeads=8,nLayers=3),
  #  dict(slidingWindow=[2,30],slidingWindowCoverage=30,nHeads=8,nLayers=6),
#    dict(slidingWindow=[2,30],slidingWindowCoverage=25,nHeads=8,nLayers=6),
#    dict(slidingWindow=[3,30],slidingWindowCoverage=15,nHeads=8,nLayers=6),
#    dict(slidingWindow=[3,30],slidingWindowCoverage=20,nHeads=8,nLayers=6),
#    dict(slidingWindow=[3,30],slidingWindowCoverage=20,nHeads=8,nLayers=6)
 #   dict(slidingWindow=[2,30],slidingWindowCoverage=30,nHidden=50,nLayers=5),
 #   dict(slidingWindow=[2,30],slidingWindowCoverage=60,nHidden=100,nLayers=5),
 #   dict(slidingWindow=[5,30],slidingWindowCoverage=60,nHidden=100,nLayers=5),
 #   dict(slidingWindow=[5,30],slidingWindowCoverage=15,nHidden=100,nLayers=5),
 #   dict(slidingWindow=[5,30],slidingWindowCoverage=30,nHidden=100,nLayers=5),
#    dict(slidingWindow=[5,30],slidingWindowCoverage=30,nHidden=100,nLayers=3),
#    dict(slidingWindow=[2,30],slidingWindowCoverage=30,nHidden=100,nLayers=5),
#    dict(slidingWindow=[5,30],slidingWindowCoverage=15,nHidden=100,nLayers=5),
#    dict(slidingWindow=[5,30],slidingWindowCoverage=30,nHidden=100,nLayers=5)
#     dict(slidingWindow=[2,10],slidingWindowCoverage=60,nHeads=8,nLayers=6)
]

if __name__=='__main__':
    for i,ba in enumerate( batch ):
        print( f"[MODEL DESCRIPTION]...................HEADS: {ba['nHeads']} | LAYERS: {ba['nLayers']} | CW: {ba['slidingWindowCoverage']} | Sliding Windows: {ba['slidingWindow']}." )
        #print( f"[MODEL DESCRIPTION]...................HIDDEN NEURONS: {ba['nHidden']} | LAYERS: {ba['nLayers']} | CW: {ba['slidingWindowCoverage']}." )
        trainer = Trainer(  )
        trainer.loadData( "/home/acmontesh/Data" )
        trainLosses, model=trainer.trainModel( "transformer",batchSize=7200,nEpochs=50,slidingWindow=ba["slidingWindow"],slidingWindowCoverage=ba["slidingWindowCoverage"],nHeads=ba["nHeads"],nLayers=ba["nLayers"],nInputs=10 )
         #trainLosses, model=trainer.trainModel( "conv_lstm",batchSize=1024,nEpochs=80,slidingWindow=ba["slidingWindow"],slidingWindowCoverage=ba["slidingWindowCoverage"],nHidden=ba["nHidden"],nLayers=ba["nLayers"] )
