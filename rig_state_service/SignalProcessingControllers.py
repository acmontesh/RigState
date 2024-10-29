# Copyright 2024 RAPID Consortium, The University of Texas at Austin, all rights reserved.
# Author: Abraham C. Montes
# ----------------------------------------------------------------------------------------
# Unauthorized reproduction, manipulation, distribution, or
# use of this code, in whole or in part, is strictly prohibited
# without the express written permission of The University of Texas.
# Contact: abraham.montes@utexas.edu
# ----------------------------------------------------------------------------------------

class SignalController:

    CONFIG_FILE_PROPOSITION_KEY     =   "propositions"
    CONFIG_FILE_FACTS_KEY           =   "facts"
    CONFIG_FILE_RULES_KEY           =   "rules"
    CONFIG_FILE_GOALS_KEY           =   "goals"
    CONFIG_FILE_TOLERANCES_KEY      =   "tolerances"
    CONFIG_FILE_RIGCHARS_KEY        =   "rigChars"

    def __init__( self,pathConfigFile ):
        self.configDict		            =	self._readConfigFile( pathConfigFile )
        self.pathConfigFile             =   pathConfigFile
    
    def getPathConfigFile( self ):
        return self.pathConfigFile

    def _readConfigFile( self,pathConfigFile ):
        import json
        with open( pathConfigFile,'r' ) as f:
            self.configDict             =  json.load( f )
        return self.configDict
    

class StateSignalsListener(SignalController):

    ROP_DIAGNOSIS_DESCRIPTION                   =   'ROP is greater than zero'
    FLOWRATE_DIAGNOSIS_DESCRIPTION              =   "Pump is on"
    ROTATION_DIAGNOSIS_DESCRIPTION              =   "Rotation is on"
    BLOCKPOSITION_DIAGNOSIS_DESCRIPTION         =   "The block is moving downwards"
    BLOCKSTATIC_DIAGNOSIS_DESCRIPTION           =   "The block is static"
    HOOKLOAD_DIAGNOSIS_DESCRIPTION              =   "The hook load is greater than the block weight"
    ROP_MNEMO                                   =   "ROP [fph]"
    FLOW_RATE_MNEMO                             =   "Flow In [gpm]"
    RPM_MNEMO                                   =   "Surface Rotation [rpm]"
    BLOCK_POSITION_MNEMO                        =   "Block Position [ft]"
    HOOK_LOAD_MNEMO                             =   "Hook Load [klb]"


    def __init__(  self, pathConfigFile,newBlockWeight=None  ):
         self.blockPositionBuffer               =       0
         self.signalsDiagnosis                  =       self._resetDiagnosisSignals(  )
         super( ).__init__( pathConfigFile )
         self.tolerances                        =       self._extractTolerances(  )
         self.rigChars                          =       self._extractRigChars( newBlockWeight )
         self.rigStateCodes                     =       self._extractRigCodes(  )
    
    def _extractTolerances( self ):
        tols                                    =       self.configDict[ self.CONFIG_FILE_TOLERANCES_KEY ]
        return tols
    
    def _extractRigChars( self,newBlockWeight ):
        chars                                   =       self.configDict[ self.CONFIG_FILE_RIGCHARS_KEY ]
        if not newBlockWeight is None:          chars[ "Block Weight" ]=newBlockWeight
        return chars
    
    def _extractRigCodes( self ):
        codes                                   =       self.configDict[ self.CONFIG_FILE_PROPOSITION_KEY ]
        codes                                   =       { val:key for key,val in codes }
        return codes

    def _resetDiagnosisSignals( self ):
        return {
                    self.ROP_DIAGNOSIS_DESCRIPTION:False,
                    self.FLOWRATE_DIAGNOSIS_DESCRIPTION:False,
                    self.ROTATION_DIAGNOSIS_DESCRIPTION:False,
                    self.BLOCKPOSITION_DIAGNOSIS_DESCRIPTION:False,
                    self.BLOCKSTATIC_DIAGNOSIS_DESCRIPTION:False,
                    self.HOOKLOAD_DIAGNOSIS_DESCRIPTION:False
                }
        
    def updateFactsRTSignals(  self,signalsDict  ):
        self.signalsDiagnosis                                           =       self._resetDiagnosisSignals(  )
        self._inspectROP( signalsDict[ self.ROP_MNEMO ] )
        self._inspectFlowRate( signalsDict[ self.FLOW_RATE_MNEMO ] )
        self._inspectRPM( signalsDict[ self.RPM_MNEMO ] )
        self._inspectBlockSpeed( signalsDict[ self.BLOCK_POSITION_MNEMO  ],signalsDict[ "Delta Time" ] )
        self._inspectHKL( signalsDict[ self.HOOK_LOAD_MNEMO ] )
        codesToWrite                                                    =       self._updateFactsFile(  )
        return self.signalsDiagnosis,codesToWrite

    def _updateFactsFile( self ):
        import json
        codesToWrite                                                    =       [  ]
        for j,key in enumerate( self.signalsDiagnosis ):
            value                                                       =       self.signalsDiagnosis[ key ]
            multiplier                                                  =       1 if value else -1
            codesToWrite.append( self.rigStateCodes[ key ] * multiplier )
        self.configDict[ self.CONFIG_FILE_FACTS_KEY ]                   =       codesToWrite
        # with open( self.pathConfigFile,'w' ) as f:
        #     json.dump(  self.configDict,f  )
        return codesToWrite

    def _inspectROP(  self,ROPValue  ):
        if ( ROPValue>self.tolerances["ROP"] ):                     self.signalsDiagnosis[self.ROP_DIAGNOSIS_DESCRIPTION]=True

    def _inspectFlowRate(  self,flowRateValue  ):
        if ( flowRateValue>self.tolerances["Flow Rate"] ):          self.signalsDiagnosis[self.FLOWRATE_DIAGNOSIS_DESCRIPTION]=True

    def _inspectRPM(  self,RPMValue  ):
        if ( RPMValue>self.tolerances["Rotation"] ):                self.signalsDiagnosis[self.ROTATION_DIAGNOSIS_DESCRIPTION]=True

    def _calculateBlockSpeed(  self,blockPositionValue,deltaTime  ):
        return ( self.blockPositionBuffer  -  blockPositionValue ) / deltaTime

    def _updateBlockPosBuffer(   self,blockPositionValue   ):
        self.blockPositionBuffer                                        =      blockPositionValue 

    def _inspectBlockSpeed(  self,blockPositionValue,deltaTime  ):
        speed                                                           =      self._calculateBlockSpeed(  blockPositionValue,deltaTime  )
        if ( abs( speed ) < self.tolerances[ "Block Position"  ] ):
            self.signalsDiagnosis[ self.BLOCKSTATIC_DIAGNOSIS_DESCRIPTION ]                     =       True
        else:
            if speed>0:                                              self.signalsDiagnosis[ self.BLOCKPOSITION_DIAGNOSIS_DESCRIPTION ]=True
        self._updateBlockPosBuffer( blockPositionValue )

    def _inspectHKL(  self,hookLoadValue  ):
        if ( ( hookLoadValue-self.rigChars[ "Block Weight" ] )>self.tolerances[ "Hook Load" ] ):          
            self.signalsDiagnosis[ self.HOOKLOAD_DIAGNOSIS_DESCRIPTION ]                        =       True


class InferenceEngine(SignalController):

    from .LogicStructs import Proposition, Rule

    def __init__(  self,pathConfigFile  ):
         self.rules		            =	[  ]
         self.facts                 =   [  ]
         self.goals                 =   [  ]
         self.goalsSet              =   set(  [x.id for x in self.goals]  )
         self.factsSet              =   set(  [x.id for x in self.facts]  )
         self.status,_              =   self._checkGoal(  )
         self.preselectedRules      =   [  ]
         super(  ).__init__( pathConfigFile )
         self._configEngineFromFile(  )

    def getGoals( self ):
        return self.goals
    
    def getGoalsDescriptions( self ):
        descriptions                =   [ x.text for x in self.goals ]
        return descriptions

    def _checkGoal(  self  ):
        intersect                   =       self.goalsSet.intersection( self.factsSet )
        if intersect:
            return True, list( intersect )
        return False, -1
    
    def addRule( self, rule: Rule ):
        self.rules.append( rule )

    def addRules(  self, *rules  ):
        for rule in rules:
            self.addRule( rule )

    def addFact(  self, fact: Proposition  ):
        self.facts.append( fact )
        self.factsSet               = set(  [x.id for x in self.facts]  )

    def addFacts( self, *facts ):
        for fact in facts:
            self.addFact(  fact  )
    
    def addGoal(  self,   goal: Proposition  ):
        self.goals.append(  goal  )
        self.goalsSet               = set( [x.id for x in self.goals] )

    def addGoals(  self, *goals  ):
        for goal in goals:
            self.addGoal(goal)

    def _preselectRules(  self  ):
        self.preselectedRules       =   [   ]
        for rule in self.rules:
            premises                =       rule.premises
            conclusions             =       rule.conclusions
            factsIds                =       list(  self.factsSet  )
            nPremisesInFacts        =   sum( [premise.id in factsIds for premise in premises] )
            nConclusionsInFacts     =   sum( [conclusion.id in factsIds for conclusion in conclusions] )
            if ( nPremisesInFacts==len( premises ) ) & ( nConclusionsInFacts<len( conclusions ) ):
                self.preselectedRules.append( rule )

    def _configEngineFromFile(  self, updateFactsOnly=False  ):
        self.propositionsBase           =  self._createListOfPropositions( self.configDict )
        self._addFactsFromFile( self.configDict )
        if not updateFactsOnly:
            self._addRulesFromFiles( self.configDict )
            self._addGoalsFromFiles( self.configDict )
    
    def updateFactsOnly( self,facts ):
        self.configDict[ self.CONFIG_FILE_FACTS_KEY ]       =   facts
        self.facts                                          =   [  ]
        self.factsSet                                       =   set(  [x.id for x in self.facts]  )
        self._addFactsFromFile( self.configDict )

    def _addRulesFromFiles(  self, configDict  ):
        from .LogicStructs import Proposition, Rule
        rules                           =       configDict[ self.CONFIG_FILE_RULES_KEY ]
        for rule in rules:
            lines                       =       rule.split('->')
            premises                    =       lines[0].split(',')
            conclusions                 =       lines[1].split(',')
            premises                    =       [ int(p) for p in premises ]
            premisesDescriptions        =       self._addPropositionsDescriptions( proposArray=premises )                        
            conclusions                 =       [ int(c) for c in conclusions ]
            conclusionsDescriptions     =       self._addPropositionsDescriptions( proposArray=conclusions )
            premises                    =       [   Proposition( premisesDescriptions[i],premises[i] ) if premises[i]>0 \
                                                    else Proposition( premisesDescriptions[i],premises[i]*-1 ).negate(  ) \
                                                    for i in range(len(premises))      ]
            conclusions                 =       [   Proposition( conclusionsDescriptions[i],conclusions[i] ) if conclusions[i]>0 \
                                                    else Proposition( conclusionsDescriptions[i],conclusions[i]*-1 ).negate(  ) \
                                                    for i in range(len(conclusions))   ]
            r                           =       Rule(  )
            r.addPremises( *premises )
            r.addConclusions( *conclusions )
            self.addRule( r )


    def _addFactsFromFile(  self, configDict  ):
        from .LogicStructs import Proposition
        facts                           =       configDict[ self.CONFIG_FILE_FACTS_KEY ]
        factsDescriptions               =       self._addPropositionsDescriptions( facts )
        f                               =       [ Proposition( factsDescriptions[i],int( facts[i] ) ) if int( facts[i] )>0 \
                                                  else Proposition( factsDescriptions[i],int( facts[i] )*-1 ).negate( ) \
                                                  for i in range(  len( facts )  )  ] 
        self.addFacts( *f )

    def _addGoalsFromFiles(  self, configDict  ):
        from .LogicStructs import Proposition
        goals                           =       configDict[ self.CONFIG_FILE_GOALS_KEY ]
        goalsDescriptions               =       self._addPropositionsDescriptions( goals )                  
        g                               =       [ Proposition( goalsDescriptions[i],goals[i] ) for i in range(  len( goals )  ) ] 
        self.addGoals( *g )

    def _addPropositionsDescriptions( self,proposArray ):
        descriptions                    =   [  ]
        proposArray                     =   [ int(x) for x in proposArray ]
        for propo in proposArray:
            for baseProp in self.propositionsBase:
                if ( baseProp.id == propo ) | ( baseProp.id == propo * -1 ):
                    descriptions.append( baseProp.text )
                    break    
        return descriptions 

    def _createListOfPropositions( self, configDict ):     
        from .LogicStructs import Proposition   
        propositionsList                =       [  ]
        bufferPropositions              =       configDict[ self.CONFIG_FILE_PROPOSITION_KEY ]
        for prop in bufferPropositions:
            propositionsList.append( Proposition( prop[1], prop[0] ) )
        return propositionsList

    def run(  self  ):
        addedFacts                  =           1
        reachedGoal                 =           -1
        while ((  addedFacts>0  ) & (  self.status==False  )):
            self._preselectRules(  )
            listConclusions         =       [  rule.getConclusions() for rule in self.preselectedRules  ]       #getting conclusions from passing rules
            listConclusions         =       [  conclusion for LC in listConclusions for conclusion in LC  ]     #unpacking
            listConclusions         =       self._filterConclusions( listConclusions )                      #filtering conclusions already in facts
            if len(  listConclusions  )==0:     
                addedFacts=0
            else:
                self.addFacts(  *listConclusions  )      
                self.status,reachedGoal     =   self._checkGoal(  )
                if self.status:
                    break
        self.status                         =   False
        if not isinstance( reachedGoal, list )  :
            #print(  "Warning! The inference engine could not reach a valid conclusion"  )            
            return None,None
        finalGoal                           =       [  x for x in self.goals if x.id == reachedGoal[0]  ][0]
        #print(  "The engine successfully reached a conclusion: {0}".format(  finalGoal.text  )  )
        return finalGoal.id, finalGoal.text
        
    def _filterConclusions(  self, conclusionsList  ):
        filteredConclusions                 = [  ]
        for conclusion in conclusionsList:
            if not (  conclusion.id in list( self.factsSet )  ):
                filteredConclusions.append( conclusion )
        return filteredConclusions