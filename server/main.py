from Trainer import Trainer

batch       = [
    # dict(slidingWindow=[5,30],slidingWindowCoverage=30,nHeads=8,nLayers=6),
    # dict(slidingWindow=[5,30],slidingWindowCoverage=60,nHeads=8,nLayers=6),
    dict(slidingWindow=[5,30],slidingWindowCoverage=120,nHeads=8,nLayers=6)
]

if __name__=='__main__':
    for i,ba in enumerate( batch ):
        print( f"[MODEL DESCRIPTION]...................HEADS: {ba['nHeads']} | LAYERS: {ba['nLayers']} | CW: {ba['slidingWindowCoverage']}." )
        trainer = Trainer(  )
        trainer.loadData( r"C:\Users\abrah\OneDrive\Doctorate Petroleum Engineering\0A. RESEARCH PROJECT\16. RIG ACTIVITY ENGINE\0. DATA\2. ANNOTATED TRAINING DATA" )
        trainLosses, model=trainer.trainModel( "transformer",batchSize=1024,nEpochs=80,slidingWindow=[5,30],slidingWindowCoverage=ba["slidingWindowCoverage"],nHeads=ba["nHeads"],nLayers=ba["nLayers"] )