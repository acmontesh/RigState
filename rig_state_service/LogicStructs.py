# Copyright 2024 RAPID Consortium, The University of Texas at Austin, all rights reserved.
# Author: Abraham C. Montes
# ----------------------------------------------------------------------------------------
# Unauthorized reproduction, manipulation, distribution, or
# use of this code, in whole or in part, is strictly prohibited
# without the express written permission of The University of Texas.
# Contact: abraham.montes@utexas.edu
# ----------------------------------------------------------------------------------------

class LogicStruct:

    def __init__(  self  ):
         self.text		        =	""
    
    def getDescription(  self  ):
        return self.text
    
    def getId(  self  ):
        return self.id
    

class Proposition(  LogicStruct  ): 

    def __init__(  self, description: str, id: int  ):
        self.id                 =       id
        self.text               =       description    
    
    def __bool__(  self  ):
        if self.id > 0:
            return True
        return False
    
    def negate(  self  ):
        self.id                 =       self.id * -1
        self.text               =       "NOT " + self.text
        return self
    

class Rule(  LogicStruct  ):
    
    def __init__(  self  ):
        self.premises           =       []
        self.conclusions        =       []
        self.text               =       ""

    def __str__(  self  ):
        return self.text
    
    def addPremise(  self, proposition: Proposition  ):
        self.premises.append(  proposition  )
        self._updateText(  )

    def addConclusion(  self, conclusion:Proposition  ):
        concBuffer              =       conclusion
        self.conclusions.append(  concBuffer  )
        self._updateText(  )

    def addPremises(  self, *premises  ):
        for premise in premises:
            self.addPremise(  premise  )

    def addConclusions(  self, *conclusions  ):
        for conclusion in conclusions:
            self.addConclusion(  conclusion  )

    def _updateText(  self  ):
        text                    =       "if: "
        for premise in self.premises: 
            text                = text + premise.text + " and "
        text                    = text[:-5]+", then: "
        for conclusion   in self.conclusions: 
            text                = text + conclusion.text
        self.text               =       text

    def getConclusions(  self  ):
        return self.conclusions