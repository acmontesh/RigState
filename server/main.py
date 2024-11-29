from Trainer import Trainer

batch       = [
    dict(slidingWindow=[2,30],slidingWindowCoverage=30,nHidden=50,nLayers=3),
    dict(slidingWindow=[5,30],slidingWindowCoverage=15,nHidden=50,nLayers=3),
    dict(slidingWindow=[5,30],slidingWindowCoverage=30,nHidden=50,nLayers=3),
    dict(slidingWindow=[2,30],slidingWindowCoverage=30,nHidden=50,nLayers=5),
    dict(slidingWindow=[5,30],slidingWindowCoverage=15,nHidden=50,nLayers=5),
    dict(slidingWindow=[5,30],slidingWindowCoverage=30,nHidden=50,nLayers=5),
    dict(slidingWindow=[2,30],slidingWindowCoverage=30,nHidden=100,nLayers=3),
    dict(slidingWindow=[5,30],slidingWindowCoverage=15,nHidden=100,nLayers=3),
    dict(slidingWindow=[5,30],slidingWindowCoverage=30,nHidden=100,nLayers=3),
    dict(slidingWindow=[2,30],slidingWindowCoverage=30,nHidden=100,nLayers=5),
    dict(slidingWindow=[5,30],slidingWindowCoverage=15,nHidden=100,nLayers=5),
    dict(slidingWindow=[5,30],slidingWindowCoverage=30,nHidden=100,nLayers=5)
]

if __name__=='__main__':
    for i,ba in enumerate( batch ):
        # print( f"[MODEL DESCRIPTION]...................HEADS: {ba['nHeads']} | LAYERS: {ba['nLayers']} | CW: {ba['slidingWindowCoverage']}." )
        print( f"[MODEL DESCRIPTION]...................HIDDEN NEURONS: {ba['nHidden']} | LAYERS: {ba['nLayers']} | CW: {ba['slidingWindowCoverage']}." )
        trainer = Trainer(  )
        trainer.loadData( "/home/acmontesh/Data" )
        trainLosses, model=trainer.trainModel( "lstm",batchSize=1024,nEpochs=80,slidingWindow=ba["slidingWindow"],slidingWindowCoverage=ba["slidingWindowCoverage"],nHidden=ba["nHidden"],nLayers=ba["nLayers"] )
