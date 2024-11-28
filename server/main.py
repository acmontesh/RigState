from Trainer import Trainer

batch       = [
    dict(slidingWindow=[5,30],slidingWindowCoverage=30,nHeads=12,nLayers=3),
    dict(slidingWindow=[5,30],slidingWindowCoverage=60,nHeads=12,nLayers=3),
    dict(slidingWindow=[5,30],slidingWindowCoverage=120,nHeads=12,nLayers=3),
    dict(slidingWindow=[5,30],slidingWindowCoverage=30,nHeads=12,nLayers=6),
    dict(slidingWindow=[5,30],slidingWindowCoverage=60,nHeads=12,nLayers=6),
    dict(slidingWindow=[5,30],slidingWindowCoverage=120,nHeads=12,nLayers=6),
]

if __name__=='__main__':
    for i,ba in enumerate( batch ):
        print( f"[MODEL DESCRIPTION]...................HEADS: {ba["nHeads"]} | LAYERS: {ba["nLayers"]} | CW: {ba["slidingWindowCoverage"]}." )
        trainer = Trainer(  )
        trainer.loadData( "/home/acmontesh/Data" )
        trainLosses, model=trainer.trainModel( "transformer",batchSize=1024,slidingWindow=[5,30],slidingWindowCoverage=60,nEpochs=80 )