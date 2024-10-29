"""
Time series annotator for the creation of well/rig activity identification engines.
Abraham C. Montes  - RAPID, UT Austin.
"""

class TimeSeries:

    WELL_ACTIVITY_COLUMN        = "Activity"
    TEMP_FILE_CUM_PERSISTANCE   = "backup_labeled_dataframe.csv"
    EDGES_SENSITIVITY           = 0.01

    def __init__( self,pathCSV,dataframe=None,preAnnotator=None,renameDimensionsJSON=None,trimDimensionsOfInterest=False,timeColumn='index',
                 timeAxisFormat="ISO8601",unitsRow=True,deleteNans=True,nanPlaceHolder=-999.25,labelColumn=None ):
        """
        inputs: 
        - pathCSV: Path to the csv file containing the time series.
        - dataframe: A TimeSeries object can also be created by passing the Pandas dataframe containing the multivariate time series. If this parameter is passed, the CSV path is ignored.
        - preAnnotator: An object to pre-annotate the time series. Typically this object applies a set of if-then clauses for an initial annotation.
                        This object must have an "annotate" method annotate(dataFrame,labelCol). Default is None.
        - renameDimensionsJSON: Path to an optional JSON file containing a hash table to rename the TS dimensions (columns). Default: None
        - trimDimensionsOfInterest: If true, only the dimensions specified in the file indicated in renameDimensionsJSON will remain.
                                    If renameDimensionsJSON is None, this parameter is simply ignored.
        - timeColumn: Name of the column in the CSV file containing the indexing variable (e.g., time). 
                      It also accepts a list of columns as input. In that case, they will be joint together before parsing as date.
                      If 'index' (default), the indexing variable will be the row number.
        - timeAxisFormat: Format of the indexing variable. Acceptable inputs:
                            * 'ISO8601' (default): Corresponds to the datetime format YYYY-mm-ddTHH:MM:SSZ e.g., 2024-08-17T11:06:31−12:00.
                            * A string containing a valid datetime format, e.g., '%m/%d/%Y %H:%M:%S'. Find more here: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior 
        - unitsRow: if True (default), the first row of the CSV file will be skipped.
        - deleteNans: if True (default), Nans are eliminated.
        - nanPlaceHolder: Number used as a placeholder for a missing or corrupt value. Default: -999.25
        - labelColumn: Name of the column in the CSV that contains pre-annotated labels. If the CSV does not contain labels, set to None. Default: None.
        """
        if dataframe is None:
            self.data,self.timeCol  =   self._loadData( pathCSV,renameDimensionsJSON,trimDimensionsOfInterest,timeColumn,
                                                    timeAxisFormat,unitsRow,deleteNans,nanPlaceHolder,labelColumn )
        else:
            self.data               = dataframe
            self.timeCol            = timeColumn
        if preAnnotator is not None:        self._preAnnotateData( preAnnotator )
        self.timeAxisFormat         = timeAxisFormat

    def _preAnnotateData( self,preAnnotator ):
        preAnnotated                = preAnnotator.annotate( self.data,self.WELL_ACTIVITY_COLUMN )
        self.data                   = preAnnotated

    def _loadData( self,pathCSV,renameDimensionsJSON,trimDimensionsOfInterest,timeColumn,timeAxisFormat,unitsRow,deleteNans,nanPlaceHolder,labelColumn ):
        import pandas as pd
        import numpy as np
        parseDates          = None
        dateFormat          = None
        timeCol             = None
        if (not timeColumn is 'index'):
            parseDates      = timeColumn
            dateFormat      = timeAxisFormat
        sRows               = [ 1 ] if unitsRow else None
        df                  = pd.read_csv( pathCSV,parse_dates=[parseDates],date_format=dateFormat,skiprows=sRows )
        if not renameDimensionsJSON is None:
            renameDict      = self._processJson( renameDimensionsJSON )
            if labelColumn is not None: renameDict[ labelColumn ] = self.WELL_ACTIVITY_COLUMN
            df              = df.rename( renameDict,axis=1 )
            if not isinstance( timeColumn,list ):
                timeCol     = renameDict[ timeColumn ] if timeColumn in list( renameDict.keys( ) ) else timeColumn
            else:
                timeCol     = '_'.join( timeColumn )
            if trimDimensionsOfInterest:
                df          = df[ list( renameDict.values( ) ) ]
            isTZAware       = pd.api.types.is_datetime64tz_dtype( df[ timeCol ] )
            if not isTZAware:
                df[ renameDict[timeCol] ]       = df[ renameDict[timeCol] ].dt.tz_localize( 'UTC' )
        else:
            if labelColumn is not None: df=df.rename( {labelColumn:self.WELL_ACTIVITY_COLUMN},axis=1 )
        if deleteNans: 
            if not nanPlaceHolder is None:
                for col in list( df.columns ):
                    df[ col ]                   = np.where( df[ col ]==nanPlaceHolder,np.nan,df[ col ] )
            nansPerCols     = list( df.isna( ).sum( ) )
            colMiss         = list( df.columns )[ np.argmax( nansPerCols ) ]
            maxPerc         = 100*(max( nansPerCols ))/df.shape[ 0 ]
            if maxPerc>10.0:
                print( f"Warning: The column with the most missing values ({colMiss}) is missing {maxPerc:.2f}% of the data. If this column is not of interest, make sure to exclude it by setting the variable trimDimensionsOfInterest to true and properly filling in a configuration JSON file" )
            df              = df.dropna( axis=0 )
        if labelColumn is None:
            df[self.WELL_ACTIVITY_COLUMN]       =   None
        return df,timeCol    
    
    def _processJson( self,renameDimensionsJSON ):
        import json
        with open( renameDimensionsJSON ) as f:
            reDict          = json.load( f )
        return reDict

    def plot( self,figWidth=6.83,figHeight=8,xTicksLastPlot=True,**kwargs ): #Default sizes are the max. figure size in SPE papers
        """
        Plots the time series. Each channel is plotted in separate panels stacked vertically.
        inputs:
        - figWidth, figHeight: Size of the figure in inches. Default values are according to SPE style guide.
        - xTicksLastPlot: If true, the x-axis' tick labels will only appear in the bottommost panel.
        *Acceptable kwargs: color,background,lineWidth,gridColor,plotFont,tickLabelsSize,axisLabelsSize,rotationXLabels,timeFormat        
        """
        #xTicksLastPlot: If true, only the last subplot shows the indexing values.
        import matplotlib.pyplot as plt
        from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
        import matplotlib.dates as mdates 
        nPlots              = self.data.shape[ 1 ] - 1
        fig,axs             = plt.subplots( nPlots,1,figsize=( figWidth,figHeight ) )
        xValues             = self.data[ self.timeCol ] if not self.timeCol is None else self.data.index
        series              = [ x for x in list( self.data.columns ) if x!=self.timeCol ]
        color               = kwargs[ "color" ] if "color" in kwargs else "teal"
        backgroundColor     = kwargs[ "background" ] if "background" in kwargs else "k"
        lineWidth           = kwargs[ "lineWidth" ] if "lineWidth" in kwargs else 0.8
        gridColor           = kwargs[ "gridColor" ] if "gridColor" in kwargs else "grey"
        plotFont            = kwargs[ "plotFont" ] if "plotFont" in kwargs else "Arial"
        tickLabelsSize      = kwargs[ "tickLabelsSize" ] if "tickLabelsSize" in kwargs else 8
        axisLabelsSize      = kwargs[ "axisLabelsSize" ] if "axisLabelsSize" in kwargs else 10
        rotationXLabels     = kwargs[ "rotationXLabels" ] if "rotationXLabels" in kwargs else 90
        timeFormat          = kwargs[ "timeFormat" ] if "timeFormat" in kwargs else None
        timeFormat          = "%m-%d-%y %H:%M" if timeFormat is None else timeFormat
        fDictTicks          = { "family":plotFont,"size":tickLabelsSize }
        fDictLabel          = { "family":plotFont,"size":axisLabelsSize }
        for j,sensor in enumerate( series ):
            if not sensor is None:
                axs[ j ].plot( xValues,self.data[ sensor ],color=color,lw=lineWidth )
                axs[ j ].set_facecolor( backgroundColor )
                axs[ j ].grid( visible=True,color=gridColor,ls="dotted" )
                axs[ j ].xaxis.set_minor_locator(AutoMinorLocator( ))
                axs[ j ].yaxis.set_minor_locator(AutoMinorLocator( ))
                axs[ j ].xaxis.set_major_formatter( mdates.DateFormatter( timeFormat ) )
                axs[ j ].set_xticklabels( axs[ j ].get_xticklabels( ),fontdict=fDictTicks )
                axs[ j ].set_yticklabels( axs[ j ].get_yticklabels( ),fontdict=fDictTicks )
                axs[ j ].tick_params( 'x',rotation=rotationXLabels )
                axs[ j ].set_xlabel( self.timeCol,fontdict=fDictLabel )
                axs[ j ].set_ylabel( sensor,fontdict=fDictLabel )
                if ( xTicksLastPlot ) & ( j<len( series )-1 ):
                     axs[ j ].set_xticklabels( [ ] )
                     axs[ j ].set_xlabel( "" )

    def split( self,segmentLength ):
        """
        Splits the time series into multiple shorter time series segments.
        inputs:
        - segmentLength: The length, in hours or indices (when no time column is set), that each segment should have. The last segment may be shorter if modulo( { length of the complete time series },segmentLength ) > 0
        output: A list with multiple Time Series objects (the segments).
        """
        import datetime as dt
        chunks              = [ ]
        initialPoint        = 0 if self.timeCol is None else self.data[self.timeCol].iloc[ 0 ]
        while True:
            if self.timeCol is not None:
                chunk       = TimeSeries( None,self.data[ (self.data[ self.timeCol ]>=initialPoint) & (self.data[ self.timeCol ]<initialPoint+dt.timedelta(hours=segmentLength)) ],
                                         preAnnotator=None,renameDimensionsJSON=None,trimDimensionsOfInterest=False,timeColumn=self.timeCol,labelColumn=self.WELL_ACTIVITY_COLUMN )
                initialPoint= initialPoint + dt.timedelta( hours=segmentLength )
            else:
                chunk       = TimeSeries( None,self.data.iloc[ initialPoint:initialPoint+segmentLength ],None,None,False,'index',labelColumn=self.WELL_ACTIVITY_COLUMN )
                initialPoint= initialPoint + segmentLength
            chunks.append( chunk )
            if self.timeCol is not None:
                if initialPoint>self.data[self.timeCol].iloc[-1]:   return chunks
            else:
                if initialPoint>self.data.shape[0]-1:               return chunks

    def save( self,path ):
        """
        Saves the current Time Series Dataframe into a CSV file. 
        input:  path, including the file name and its extension, where the time series should be stored.
        """
        self.data.to_csv( path,index=False )

    def plotSummaryLabels( self,figW=6.83,figH=5,**kwargs ):
        import matplotlib.pyplot as plt
        color       = 'magenta' if "color" not in kwargs else kwargs["color"]
        lineWidth   = 'magenta' if "lineWidth" not in kwargs else kwargs["lineWidth"]
        background  = 'black' if "background" not in kwargs else kwargs["background"]
        fig,ax      = plt.subplots( figsize=(figW,figH) )
        x           =  self.data.index if self.timeCol is None else self.data[ self.timeCol ]
        ax.plot( x, self.data[self.WELL_ACTIVITY_COLUMN], color=color, lw=lineWidth )
        ax.invert_yaxis( )
        ax.grid( visible=True,color='lightgrey',ls='dotted', which='both' )
        ax.set_ylabel( "Label" )
        ax.set_xlabel( "Time/Index" )
        ax.set_facecolor( background )

    def annotate( self,hoursPerPlot=4,rangeIdxPerPlot=1000,figWidth=15,figHeight=8,activityCodes=None,cumulativePersistance=True,**kwargs ):
        """
        Allows the interactive annotation of the time series object
        inputs:
        - hoursPerPlot: The number of hours that will appear in each plot. Default is 4h.
        - rangeIdxPerPlot: If self.timeCol is set to None, then each plot will contain this number of indices for annotation.
        - figWidth, figHeight: Size of the figure in inches. 
        - activityCodes: List of activity codes to have in the activity "brush" palette (i.e., the ones that will be available to annotate the data).
                        Default is None. If set to None, it will simply use the pre-existing annotations (e.g., the ones assigned by the preAnnotator).
        - cumulativePersistance: Defines whether the dataframe is continously persisted on the disk each time a new time interval is labeled.
                                *** Is it highly recommended to keep this as True.
        *Acceptable kwargs: palette,background,lineWidth,alpha,selectionAlpha,plotFont,tickLabelsSize,axisLabelsSize,rotationXLabels,timeFormat,tickLength,yAxisWidth
        WARNING: This function requires using the Qt matplotlib backend. To activate it, ensure you properly install PyQt5 through 'pip install PyQt5' and use the magic command '%matplotlib qt' before calling this function.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button
        import matplotlib.dates as mdates
        from matplotlib.dates import date2num, num2date
        from matplotlib.ticker import AutoMinorLocator
        import pandas as pd
        import datetime as dt
        import matplotlib.cm as cm
        import sys
        import ctypes
        try:
            cmap        = cm.get_cmap('tab20') if not "palette" in kwargs else kwargs[ "palette" ]
        except:
            print( f"Error: The provided palette is not a valid colormap." )
            sys.exit( )
        if activityCodes is not None:
            activities      = activityCodes
        elif not (self.data[ self.WELL_ACTIVITY_COLUMN ].values==None).all( ):
            activities      = list(  np.unique( np.array( [ x for x in self.data[ self.WELL_ACTIVITY_COLUMN ].values if x is not None ] ) )  )
        else:
            MessageBox = ctypes.windll.user32.MessageBoxW
            MessageBox(  None, f'There are no pre-existing labels in this time series. You can annotate the series from scratch, but you must then provide the list of possible labels when calling the annotate function', 'No Labels to Annotate', 0x30  ) 
            sys.exit( )
        colorDict           = dict( zip( activities,[ cmap( j ) for j in range( len( activities ) ) ] ) )
        backgroundColor     = kwargs[ "background" ] if "background" in kwargs else "k"
        lineWidth           = kwargs[ "lineWidth" ] if "lineWidth" in kwargs else 0.8
        yAxisWidth          = kwargs[ "yAxisWidth" ] if "yAxisWidth" in kwargs else 2.0
        alpha               = kwargs[ "alpha" ] if "alpha" in kwargs else 0.4
        selectionAlpha      = kwargs[ "selectionAlpha" ] if "selectionAlpha" in kwargs else 0.9       
        plotFont            = kwargs[ "plotFont" ] if "plotFont" in kwargs else "Arial"
        tickLabelsSize      = kwargs[ "tickLabelsSize" ] if "tickLabelsSize" in kwargs else 8
        axisLabelsSize      = kwargs[ "axisLabelsSize" ] if "axisLabelsSize" in kwargs else 10
        rotationXLabels     = kwargs[ "rotationXLabels" ] if "rotationXLabels" in kwargs else 90
        timeFormat          = kwargs[ "timeFormat" ] if "timeFormat" in kwargs else None
        tickLength          = kwargs[ "tickLength" ] if "tickLength" in kwargs else 6
        timeFormat          = "%m-%d-%y %H:%M" if timeFormat is None else timeFormat
        fDictTicks          = { "family":plotFont,"size":tickLabelsSize }
        fDictLabel          = { "family":plotFont,"size":axisLabelsSize,"weight":"bold" }
        sensors     =   list( set( list( self.data.columns ) ) - set( [self.timeCol,self.WELL_ACTIVITY_COLUMN] ) )
        xminGlobal  =   0 if self.timeCol is None else self.data[ self.timeCol ].iloc[ 0 ]
        xmaxGlobal  =   rangeIdxPerPlot if self.timeCol is None else xminGlobal+dt.timedelta( hours=hoursPerPlot )
        selectionTol=   rangeIdxPerPlot/100 if self.timeCol is None else (4/2400)

        global selectedRectangle, currentAct, pressEvent

        rectHeightRef       = self.data[ sensors[ 0 ] ].max( ) - self.data[ sensors[ 0 ] ].min( )
        rectYMinRef         = self.data[ sensors[ 0 ] ].min( )
        selectedRectangle   = None
        pressEvent          = None
        currentAct          = list( colorDict.keys( ) )[ 0 ]
        rectangles          = [  ]
        existingActs        = [  ]
        currentBrughText    = None

        def onPress( event ):
            global selectedRectangle, currentAct, pressEvent
            pressEvent                          = event
            if event.inaxes is not None:
                for j,rect in enumerate( rectangles ):
                    x0, y0                      = rect.get_xy(  )
                    width                       = rect.get_width(  )
                    if selectedRectangle is not None:
                            selectedRectangle[ 0 ].set_alpha( alpha )
                            selectedRectangle   = None
                    if abs(event.xdata - x0) < self.EDGES_SENSITIVITY:       
                        selectedRectangle       = (rect, 'left',j)
                        break
                    elif abs(event.xdata - (x0 + width)) < self.EDGES_SENSITIVITY:   #selectionTol
                        selectedRectangle       = (rect, 'right',j)
                        break
                    elif event.xdata>x0 and event.xdata<(x0+width):                        
                        selectedRectangle       = ( rect, 'sel',j )
                        rect.set_alpha( selectionAlpha )      
                        break
                if selectedRectangle is not None:   currentAct                      =   existingActs[  selectedRectangle[2]  ]

        def onMotion(event):
            global selectedRectangle, currentAct, pressEvent
            if selectedRectangle is None or event.xdata is None:
                return
            rect, side,_        = selectedRectangle
            x0, y0              = rect.get_xy( )
            width               = rect.get_width( )
            newX0               = x0
            newWidth            = width

            if side == 'left':
                newX0 = event.xdata
                newWidth = (x0 + width) - newX0
            elif side == 'right':
                newWidth = event.xdata - x0
            
            if newWidth > 0:
                otherRectangles = [ re for re in rectangles if re is not rect ]
                otherRectanglesX0=[ re.get_xy(  )[ 0 ] for re in otherRectangles ]
                otherRectanglesXMax=[ re.get_width(  ) + re.get_xy(  )[ 0 ] for re in otherRectangles ]
                diffX0          = np.array( [ x0 + width - oX0 for oX0 in otherRectanglesX0 ] )
                diffXMax        = np.array( [ x0 - oXMax for oXMax in otherRectanglesXMax ] )
                rightRectIdx    = np.argmax( np.where( diffX0>0,-np.inf,diffX0) ) if (diffX0<=0).sum( )>0 else None
                leftRectIdx     = np.argmin( np.where( diffXMax<0,np.inf,diffXMax) ) if (diffXMax>=0).sum( )>0 else None
                if rightRectIdx is not None:
                    rightRX0        = otherRectanglesX0[ rightRectIdx ]
                else:
                    rightRX0        = xmaxGlobal if self.timeCol is None else date2num( xmaxGlobal )
                if leftRectIdx is not None:            
                    leftRXmax       = otherRectanglesXMax[ leftRectIdx ]
                else:
                    leftRXmax       = xminGlobal if self.timeCol is None else date2num( xminGlobal )
                if side == 'left' and newX0<leftRXmax:
                    newX0           = leftRXmax
                    newWidth        = (x0 + width) - newX0
                elif side == 'right' and (newX0 + newWidth) > rightRX0:
                    newWidth        = rightRX0 - newX0            
                
                if newWidth > 0:
                    rect.set_x(newX0)
                    rect.set_width(newWidth)
                    # self.data[ self.WELL_ACTIVITY_COLUMN ]   =    np.where( (self.data[self.timeCol]>=num2date( newX0 )) & (self.data[self.timeCol]<num2date( newX0+newWidth )), currentAct, self.data[ self.WELL_ACTIVITY_COLUMN ] )
                plt.draw(  )

        def onRelease( event ):
            global selectedRectangle, currentAct, pressEvent
            if selectedRectangle is not None:
                rect, side,_            = selectedRectangle
                x0, y0                  = rect.get_xy(  )
                width                   = rect.get_width(  )
                xmin                    = x0
                xmax                    = x0 + width
                if selectedRectangle[1]=='left' or selectedRectangle[1]=='right':
                    selectedRectangle   = None

        def onKey( event ):
            global selectedRectangle, currentAct, pressEvent
            if event.key == 'delete' and selectedRectangle is not None:
                rectToRemove            = selectedRectangle[0]
                x0,_                    = rectToRemove.get_xy( )
                xMax                    = x0 + rectToRemove.get_width( )
                x0Date,xMaxDate     = ( num2date( x0 ), num2date( xMax ) )
                if self.timeCol is not None:
                        self.data[ self.WELL_ACTIVITY_COLUMN ]  =   np.where( (self.data[self.timeCol]>=x0Date) & (self.data[self.timeCol]<xMaxDate),None,self.data[ self.WELL_ACTIVITY_COLUMN ] )
                else:
                    self.data[ self.WELL_ACTIVITY_COLUMN ].iloc[ x0:x0+newWidth ]   =   None
                del rectangles[ selectedRectangle[2] ]
                del existingActs[ selectedRectangle[2] ]
                rectToRemove.remove( )
                selectedRectangle       = None
                plt.draw(  )
            elif event.key == 'n':  
                if event.inaxes is not None:
                    x0                  = event.xdata
                    otherRectanglesX0   = [ re.get_xy(  )[0] for re in rectangles ]
                    rectangleViolations = [ 1 if (x0>=re.get_xy( )[0]) & (x0<=(re.get_xy( )[0] + re.get_width( ))) else 0 for re in rectangles ]
                    if sum( rectangleViolations )>0:
                        MessageBox = ctypes.windll.user32.MessageBoxW
                        MessageBox(  None, f'It is not allowed to create a new activity inside another activity. Please move the mouse outside existing annotations or modify the existing annotations prior to attempting to create a new one at this location', 'Forbidden Annotation', 0x30  ) 
                    else:
                        applicableRX0       = [  rx for rx in otherRectanglesX0 if rx > x0  ]
                        rightRX0            = min(applicableRX0) if len( applicableRX0 )>0 else date2num( xmaxGlobal )
                        newWidth            = rightRX0 - x0
                        x0Date,xMaxDate     = ( num2date( x0 ), num2date( rightRX0 ) )                    
                        newRect             = Rectangle(( x0, rectYMinRef ), newWidth, rectHeightRef, color=colorDict[ currentAct ], alpha=alpha)
                        rectangles.append( newRect )
                        existingActs.append( currentAct )
                        ax.add_patch( newRect )
                        plt.draw(  )
            elif event.key == 'escape':  
                if event.inaxes is not None:
                    plt.close( fig )

        def setCurrentAct( act ):
            global currentAct
            currentAct                  = act
            currentBrughText.set_text( f"Selected activity brush: {currentAct}" )
            plt.draw(  )

        paletteButtons     = [  ]      

        while True:           
            dataExcerpt     = self.data[ (self.data[ self.timeCol ]>=xminGlobal) & (self.data[ self.timeCol ]<xmaxGlobal) ]
            if dataExcerpt.shape[ 0 ]==0:       break
            fig, ax             = plt.subplots( figsize=( figWidth,figHeight ) )
            rectangles          = [  ]
            existingActs        = [  ]
            x               = dataExcerpt[ self.timeCol ] if self.timeCol is not None else np.arange( xminGlobal,xmaxGlobal,1 )
            xmaxPlot        =   date2num( x.max( ) ) if self.timeCol is not None else x.max(  )
            xminPlot        =   date2num( x.min( ) ) if self.timeCol is not None else x.min(  )
            for j,sensor in enumerate( sensors ): 
                y           =    dataExcerpt[ sensor ]                
                yMin        =    self.data[ sensor ].min( ) #To ensure all plots have the same scale for each sensor.
                yMax        =   self.data[ sensor ].max( )
                if j==0:        
                    yMinBuff                = yMin if yMin!=yMax else yMin-1
                    yHeightBuff             = yMax-yMin if yMin!=yMax else 2 
                    yMax                    = yMax if yMin!=yMax else yMin+1                   
                    ax.plot(  x, y, color=cmap( j ),lw=lineWidth  )
                    ax.set_ylim( yMin,yMax )
                    ax.set_ylabel( f"{sensor}",fontdict=fDictLabel )
                    ax.yaxis.set_minor_locator( AutoMinorLocator(  ) )
                    ax.tick_params( 'y',which='major',length=tickLength  )
                    ax.spines['left'].set_linewidth( yAxisWidth )
                    ax.tick_params( 'x',rotation=rotationXLabels )
                    ax.xaxis.set_major_formatter( mdates.DateFormatter( timeFormat ) )
                    ax.set_xticklabels( ax.get_xticklabels( ),fontdict=fDictTicks )
                    ax.set_yticklabels( ax.get_yticklabels( ),fontdict=fDictTicks )
                    ax.spines['left'].set_color( cmap( j ) )
                    ax.set_facecolor( backgroundColor )
                    currentBrughText        =   ax.text( xminGlobal,yMinBuff-(yMax-yMinBuff)/5,f"Selected activity brush: {currentAct}",fontdict=fDictLabel )
                else:                    
                    ax2 = ax.twinx( )
                    ax2.plot( x,y,color=cmap( j ),lw=lineWidth )
                    ax2.set_ylim( yMin,yMax )
                    ax2.spines['right'].set_position( ('axes', 0.9+j/10) )
                    ax2.spines['right'].set_color( cmap( j ) )
                    ax2.spines['right'].set_linewidth( yAxisWidth )
                    ax2.yaxis.set_ticks_position( 'right' )
                    ax2.yaxis.set_label_position( 'right' )
                    ax2.yaxis.set_minor_locator( AutoMinorLocator(  ) )
                    ax2.set_ylabel( sensor,fontdict=fDictLabel )
                    ax2.tick_params( 'y',which='major',length=tickLength )
                    ax2.set_yticklabels( ax2.get_yticklabels( ),fontdict=fDictTicks )
                    ax2.set_facecolor( backgroundColor )                    


            if not ( dataExcerpt[ self.WELL_ACTIVITY_COLUMN ].values==None ).all( ):
                preAnno                 = dataExcerpt[ self.WELL_ACTIVITY_COLUMN ].values
                preAnno                 = np.where( preAnno==None,-1,preAnno )
                preAnnoDiff             = np.diff( preAnno )
                idxChanges              = np.argwhere( preAnnoDiff!=0 )
                if idxChanges.shape[ 0 ]==0:
                    idxChanges          = np.append( idxChanges,preAnno.shape[ 0 ]//2 )
                initialXMin             = xminPlot
                for k,chPoint in np.ndenumerate( idxChanges ):
                    thisAct             = preAnno[ chPoint ]
                    if thisAct!=-1:
                        thisColor           = colorDict[ thisAct ]
                        delta               = date2num( dataExcerpt[ self.timeCol ].iloc[ chPoint ] ) - initialXMin
                        rectangles.append( Rectangle( ( initialXMin,yMinBuff ),delta,yHeightBuff ,color=thisColor,alpha=alpha ) )
                        existingActs.append( thisAct )
                    initialXMin             = date2num( dataExcerpt[ self.timeCol ].iloc[ chPoint ] ) if self.timeCol is not None else chPoint
                    if chPoint==idxChanges[-1] and chPoint<preAnnoDiff.size-1:
                        if preAnno[ chPoint+1 ]!=-1:
                            thisAct             = preAnno[ chPoint+1 ]
                            thisColor           = colorDict[ thisAct ]
                            finalXMax           = xmaxGlobal if self.timeCol is None else date2num( xmaxGlobal )
                            delta               = finalXMax - initialXMin
                            rectangles.append( Rectangle( ( initialXMin,yMinBuff ),delta,yHeightBuff ,color=thisColor,alpha=alpha ) )
                            existingActs.append( thisAct )
                for re in rectangles:
                    ax.add_patch( re )

            activityBrushes             = [  ]
            for idx, act in enumerate( colorDict ):
                button_ax               = fig.add_axes([ 0.02, 0.9 - idx * 0.05, 0.05, 0.04 ])
                button                  = Button(button_ax, act, color=colorDict[ act ], hovercolor=colorDict[ act ])
                activityBrushes.append(  button  )

            for button, act in zip( activityBrushes, activities ):
                button.on_clicked( lambda event, act=act: setCurrentAct( act ) )  

            fig.canvas.mpl_connect(   'button_press_event', onPress   )
            fig.canvas.mpl_connect(   'motion_notify_event', onMotion   )
            fig.canvas.mpl_connect(   'button_release_event', onRelease   )
            fig.canvas.mpl_connect(   'key_press_event', onKey   )
            plt.tight_layout( w_pad=1 )
            plt.show(  )            
            MessageBox = ctypes.windll.user32.MessageBoxW
            MessageBox(  None, 'Click to continue after finishing annotating the current plot', 'Continue', 0  ) 
            plt.close(  )

            if self.timeCol is not None:
                self.data[ self.WELL_ACTIVITY_COLUMN ]  =   np.where( (self.data[self.timeCol]>=xminGlobal) & (self.data[self.timeCol]<xmaxGlobal),None,self.data[ self.WELL_ACTIVITY_COLUMN ] )
            else:
                self.data[ self.WELL_ACTIVITY_COLUMN ].iloc[ xminGlobal:xmaxGlobal ]   =   None
            for j,re in enumerate( rectangles ):
                x0                      = re.get_xy( )[0]
                xmax                    = x0 + re.get_width(  )
                x0                      = x0 if self.timeCol is None else num2date( x0 )
                xmax                    = xmax if self.timeCol is None else num2date( xmax )
                thisAct                 = existingActs[ j ]
                if self.timeCol is not None:
                    self.data[ self.WELL_ACTIVITY_COLUMN ]  =   np.where( (self.data[self.timeCol]>=x0) & (self.data[self.timeCol]<xmax),thisAct,self.data[ self.WELL_ACTIVITY_COLUMN ] )
                else:
                    self.data[ self.WELL_ACTIVITY_COLUMN ].iloc[ x0:xmax ]   =   thisAct
            bufferDataFrame             = self.data[ (self.data[self.timeCol]<xmaxGlobal) ]
            bufferDataFrame.to_csv( self.TEMP_FILE_CUM_PERSISTANCE,index=False )
            xminGlobal                  = xmaxGlobal
            xmaxGlobal                  = xminGlobal+rangeIdxPerPlot if self.timeCol is None else xminGlobal+dt.timedelta( hours=hoursPerPlot )