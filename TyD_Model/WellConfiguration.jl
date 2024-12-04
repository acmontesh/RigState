#=
Copyright 2024 The University of Texas at Austin, RAPID Consortium
Created by: Abraham C. Montes, M.Sc., Ph.D. Student
Description: _
License: Apache 2.0
=#


module WellConfiguration

struct SurveyStation
    depth::            Union{ Integer,AbstractFloat }   #depths of survey stations, in ft
    inclination::      AbstractFloat   #Inclination values, in degrees
    azimuth::          AbstractFloat   #Azimuth values, in degrees
end

struct Survey
    stations::          Vector{ SurveyStation }
end

mutable struct BHAComponent
    componentNo::           Integer
    description::           AbstractString  #A custom description (or name) of the element
    OD::                    AbstractFloat #External diameter of the element in inches
    ID::                    AbstractFloat #Internal diameter of the element in inches
    length::                Union{ AbstractFloat,Integer }   #Length of the element in ft
    nominalWeight::         Union{ AbstractFloat,Integer }  #weight in pounds per ft without bouyancy effects
    MDBottomOfElement::     Union{ Nothing,Integer,AbstractFloat }  #The position of the lower end of the element
    MDTopOfElement::        Union{ Nothing,Integer,AbstractFloat }  #The position of the top of the element
    γ::                     Union{ AbstractFloat }
end

mutable struct BHA
    components::        Vector{ BHAComponent }
    bottomToTop::       Bool #Is element No. 1 located at the bottom of the drillstring?
end

mutable struct WellboreSection
    sectionNo::         Integer
    description::       AbstractString
    TD::                Union{ Nothing,Integer,AbstractFloat }
    ID::                Union{ Nothing,AbstractFloat }  
    μ::                 Union{ Nothing,AbstractFloat }
    isμFixed::          Bool
end

mutable struct Wellbore
    sections::          Vector{ WellboreSection }
    ToptoBottom::       Bool 
    TD::                Union{ Nothing,Integer,AbstractFloat }
end

mutable struct Fluid
    ρᵢ::                 AbstractFloat
    ρₒ::                 AbstractFloat
    μₚ::                AbstractFloat
    τₒ::                AbstractFloat
    isWaterBased::      Bool
end


end