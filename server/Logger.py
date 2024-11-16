import logging

class LoggerDev:

    def __init__( self,verbosity=2 ):
      for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
      logging.basicConfig(   format='[%(levelname)s]......... %(message)s', level=logging.DEBUG   )
      self.verbosity      =   verbosity

    def setVerbosity( self,verbosity ):
        self.verbosity      = verbosity
        logging.info( f"Verbosity has been set to {verbosity}" )

    def dbgMsg( self,msg=None ):
        if (msg is not None) and (self.verbosity>1):
            logging.debug(  msg  )

    def infoMsg( self,msg=None ):
        if (msg is not None) and (self.verbosity>1):
            logging.info(  msg  )

    def warningMsg( self,msg=None ):
        if (msg is not None) and (self.verbosity>0):
            logging.warning(  msg  )

    def errorMsg( self,msg=None ):
        if (msg is not None) and (self.verbosity>-1):
            logging.error(  msg  )
