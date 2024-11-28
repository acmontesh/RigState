from Trainer import Trainer

if __name__=='__main__':
    trainer     = Trainer(  )
    trainer = Trainer(  )
    trainer.loadData( "/home/acmontesh/Data" )
    trainLosses, model=trainer.trainModel( "transformer",batchSize=1024,slidingWindow=[5,30],slidingWindowCoverage=30,nEpochs=80 )
