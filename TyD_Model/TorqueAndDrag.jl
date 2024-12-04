module TnD

using Pkg
Pkg.add(["CSV", "DataFrames", "StaticArrays","DifferentialEquations","ForwardDiff","Sundials","JSON"])
include("WellConfiguration.jl")
using CSV
using DataFrames
using StaticArrays
using DifferentialEquations
using ForwardDiff
using Sundials
using Printf
using JSON
using .WellConfiguration

function readSurvey( path::String )
    df = CSV.read( path, DataFrame )
    sᵢ = df[!,:MD]*1.0
    ϕᵢ = df[!,:INCLINATION]
    θᵢ = df[!,:AZIMUTH]
    sᵢ,ϕᵢ,θᵢ
end

function calculateTrajectory( sᵢ,ϕᵢ,θᵢ )
    n       = length(sᵢ)
    λᵢ      =  Matrix{Float64}(undef, n, 3)
    λᵢ[:, 1] = sin.(deg2rad.(ϕᵢ)).*sin.(deg2rad.(θᵢ))
    λᵢ[:, 2] = sin.(deg2rad.(ϕᵢ)).*cos.(deg2rad.(θᵢ))
    λᵢ[:, 3] = cos.(deg2rad.(ϕᵢ))    
    hᵢ      = (circshift( sᵢ,-1 ) .- sᵢ)[1:end-1]
    uᵢ      = 2*( circshift( hᵢ,1 ) .+ hᵢ )[2:end]
    vᵢ      = Matrix{Float64}(undef,n-2,3)
    for i in 2:length(sᵢ)-1
        f₁          = ( λᵢ[i+1,:].-λᵢ[i,:] ) ./ hᵢ[i]
        f₂          = ( λᵢ[i,:]-λᵢ[i-1,:] ) ./ hᵢ[i-1]
        vᵢ[i-1,:]   = 6*(f₁-f₂)
    end
    A       =zeros(n-2,n-2)
    ζ₁      =uᵢ[1]+hᵢ[1]+(hᵢ[1]^2)/hᵢ[2]
    ζ₂      =hᵢ[2]-(hᵢ[1]^2)/hᵢ[2]
    ζ₃      =hᵢ[end-1]-(hᵢ[end]^2)/hᵢ[end-1]
    ζ₄      =uᵢ[end]+hᵢ[end]+(hᵢ[end]^2)/hᵢ[end-1]
    A[1,1]  =ζ₁
    A[1,2]  =ζ₂
    A[n-2,n-3]=ζ₃
    A[n-2,n-2]=ζ₄
    for i in 2:n-3
        A[i,i-1]=hᵢ[i]
        A[i,i]  =uᵢ[i]
        A[i,i+1]=hᵢ[i+1]
    end
    z           =Matrix{Float64}(undef,n-2,3)
    for i in 1:3
        bᵢ      =vᵢ[:,i]
        problem =LinearProblem(A, bᵢ)
        sol     =solve( problem,nothing )
        z[:,i]  =sol.u
    end
    z₀          = z[1,:] .- hᵢ[1]*((z[2,:].-z[1,:]) ./ hᵢ[2])
    zₙ          = z[end,:] .+ hᵢ[end]*((z[end,:].-z[end-1,:]) ./ hᵢ[end-1])
    Z           =Matrix{Float64}(undef,n,3)
    Z[1,:]      =z₀
    Z[2:n-1,:]  =z
    Z[n,:]      =zₙ
    Aᵢ          =λᵢ[1:end-1,:]
    Bᵢ          =( circshift(λᵢ,-1)[1:end-1,:] .- λᵢ[1:end-1,:]  )./hᵢ  .-  (1/6)*hᵢ.*circshift( Z,-1 )[1:end-1,:]  .-  (1/3)*hᵢ.*Z[1:end-1,:]
    Cᵢ          =0.5*Z[1:end-1,:]
    Dᵢ          =(circshift( Z,-1 )[1:end-1,:].-Z[1:end-1,:]) ./ (6*hᵢ)
    Aᵢ,Bᵢ,Cᵢ,Dᵢ,Z
end

function T(s::Union{AbstractFloat,Integer},sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ)
    isᵢ     =searchsortedfirst( sᵢ,s )
    Δsᵢ     =s-sᵢ[isᵢ-1]
    Tᵪ      =Aᵢ[isᵢ-1,:] .+ Δsᵢ*Bᵢ[isᵢ-1,:]  .+  (Δsᵢ^2)*Cᵢ[isᵢ-1,:]  .+  (Δsᵢ^3)*Dᵢ[isᵢ-1,:]
    Tᵪ,isᵢ,Aᵢ[isᵢ-1,:],Bᵢ[isᵢ-1,:],Cᵢ[isᵢ-1,:],Dᵢ[isᵢ-1,:]
end

function Tₜ(s::Union{AbstractFloat,Integer},sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ)
    _,isᵢ,_,_,_,_=T( s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
    wₚ      =[0.,0.,sᵢ[2]]
    for j in 2:isᵢ-1
        _,_,A,B,C,D=T( sᵢ[j],sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
        wₚ[1]=wₚ[1]+(hᵢ[j]*A[1]+((hᵢ[j]^2)/2)*B[1]+((hᵢ[j]^3)/3)*C[1]+((hᵢ[j]^4)/4)*D[1])
        wₚ[2]=wₚ[2]+(hᵢ[j]*A[2]+((hᵢ[j]^2)/2)*B[2]+((hᵢ[j]^3)/3)*C[2]+((hᵢ[j]^4)/4)*D[2])
        wₚ[3]=wₚ[3]+(hᵢ[j]*A[3]+((hᵢ[j]^2)/2)*B[3]+((hᵢ[j]^3)/3)*C[3]+((hᵢ[j]^4)/4)*D[3])
    end
    wₚ
end

function readJson( jsonPath::String )
    jsonString      = read(jsonPath, String)
    parsedData      = JSON.parse(jsonString)
    parsedData
end

function readBHA( jsonPath::String )
    parsedData      = readJson( jsonPath )
    compArray       = [ ]
    currentMDTop    = nothing
    currentMDBottom = nothing
    compLength      = nothing
    for (i,comDict) in enumerate( parsedData["components"] )
        if i==length( parsedData["components"] )
            compLength      = currentMDTop
        else
            compLength  = comDict["length"]
        end
        if i==1
            currentMDBottom =parsedData["MDBottom"]
        else
            currentMDBottom =currentMDTop
        end
        currentMDTop        =currentMDBottom-compLength
        component           = WellConfiguration.BHAComponent( i,comDict["name"],comDict["OD"],comDict["ID"],compLength,comDict["adjustedNominalWeight"],currentMDBottom,currentMDTop,comDict["youngModulus"] )
        push!( compArray,component )
    end
    BHAₒ                    = WellConfiguration.BHA( compArray,true )
    BHAₒ
end

function readWellbore( jsonPath::String )
    parsedData      = readJson( jsonPath )
    sectArray       = [ ]
    for (i,sect) in enumerate( vcat( parsedData["previousCasings"],parsedData["openHoleSections"] ) )
        push!(sectArray,WellConfiguration.WellboreSection( i,sect["name"],sect["bottomMD"],sect["ID"],sect["frictionCoefficient"],sect["isFFFixed"] ))
    end
    Wellbore                    = WellConfiguration.Wellbore( sectArray,true,parsedData["sectionTD"] )
    Wellbore
end

function readFluid( jsonPath::String )
    p               = readJson( jsonPath )
    f               = WellConfiguration.Fluid( p["MWIn"],p["MWOut"],p["plasticViscosity"],p["yieldPoint"],p["isWaterBased"] )
    f
end

function TE( s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
    isᵢ     =searchsortedfirst( sᵢ,s )
    Δsᵢ     =s-sᵢ[isᵢ-1]
    Tᵪ      =Aᵢ[isᵢ-1,:] .+ Δsᵢ*Bᵢ[isᵢ-1,:]  .+  (Δsᵢ^2)*Cᵢ[isᵢ-1,:]  .+  (Δsᵢ^3)*Dᵢ[isᵢ-1,:]
    Tᵪ[1]
end

function TN( s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
    isᵢ     =searchsortedfirst( sᵢ,s )
    Δsᵢ     =s-sᵢ[isᵢ-1]
    Tᵪ      =Aᵢ[isᵢ-1,:] .+ Δsᵢ*Bᵢ[isᵢ-1,:]  .+  (Δsᵢ^2)*Cᵢ[isᵢ-1,:]  .+  (Δsᵢ^3)*Dᵢ[isᵢ-1,:]
    Tᵪ[2]
end

function TT( s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
    isᵢ     =searchsortedfirst( sᵢ,s )
    Δsᵢ     =s-sᵢ[isᵢ-1]
    Tᵪ      =Aᵢ[isᵢ-1,:] .+ Δsᵢ*Bᵢ[isᵢ-1,:]  .+  (Δsᵢ^2)*Cᵢ[isᵢ-1,:]  .+  (Δsᵢ^3)*Dᵢ[isᵢ-1,:]
    Tᵪ[3]
end

function ϕ( s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
    if s==0.
        return 0.0
    end
    x       = TT(s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ)
    if x>1.
        x=1.
    end
    return acos(x)
end

function cosϕ( s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
    if s==0.
        return 1.0
    end
    x       = TT(s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ)
    if x>1.
        x=1.
    end
    return x
end

function θ( s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
    if s==0.
        return 0.0
    end
    return atan(TE(s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ)/TN(s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ))
end

function dϕ( s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
    if s==0.
        return 0.0
    end
    ∂ϕ  =   x -> ϕ( x,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
    return ForwardDiff.derivative(∂ϕ,s)
end

function calculateμ( s,wellbore::WellConfiguration.Wellbore )
    for sect in wellbore.sections
        if sect.TD>=s
            return sect.μ
        end
    end
end

function locateBHAInHole( MDBottom,bha::WellConfiguration.BHA )
    ΔZ                          = bha.components[1].MDBottomOfElement - MDBottom
    mdTopsCorrected             = [ c.MDTopOfElement - ΔZ for c in bha.components ]
    mdBottomsCorrected          = [ c.MDBottomOfElement - ΔZ for c in bha.components ]
    iElement                    = findfirst( x->x<=0,mdTopsCorrected )
    newComponents               = [ ]
    for j in 1:1:iElement
        c                       = bha.components[j]
        λ                       = c.length
        Zₜ                      = mdBottomsCorrected[j] - λ
        if j==iElement
            λ                   = mdBottomsCorrected[j]
            Zₜ                  = 0.0
        end
        push!( newComponents,WellConfiguration.BHAComponent( j,c.description,c.OD,c.ID,λ,c.nominalWeight,mdBottomsCorrected[j],Zₜ,c.γ ) )
    end
    WellConfiguration.BHA( newComponents,true )
end

function getCHμ( s,wellbore::WellConfiguration.Wellbore )
    i       = findfirst( x -> x.TD>s,wellbore.sections )
    wellbore.sections[i].μ,wellbore.sections[i].isμFixed
end

function wᵦ( s,fluid::WellConfiguration.Fluid,drillstring::WellConfiguration.BHA )
    ρᵢ      = fluid.ρᵢ
    ρₒ      = fluid.ρₒ
    bf      = ( 65.5-ρᵢ ) / 65.5
    if s==0.
        return 0.
    else
        MDTops  = [ c.MDTopOfElement for c in drillstring.components ]
        reversedTops=reverse(MDTops)
        i       = searchsortedfirst( reversedTops,s ) - 1
        w       = drillstring.components[end-i+1].nominalWeight
        Dᵢ      = drillstring.components[end-i+1].ID
        Dₒ      = drillstring.components[end-i+1].OD
        bf      = 0.0408*( ρᵢ*Dᵢ^2 - ρₒ*Dₒ^2 )
        return w + bf
    end
end

function dθ( s,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
    if s==0.
        return 0.0
    end
    ∂𝜃  =   x -> θ( x,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
    return ForwardDiff.derivative(∂𝜃,s)
end

function inferμ( s,μOH, wellbore::WellConfiguration.Wellbore )
    μ,isFixed      = getCHμ( s,wellbore )
    if !isFixed
        return μOH
    else
        μ         = μOH>0 ? μ : -1*μ
        return μ
    end
end

function FₜSA( u,params,t )
    μOH,fluid,bha,wellbore,sᵢ,A,B,C,D = params
    Fₜ            = u[1]
    s             = t[1]
    dFₜ           = -wᵦ( s,fluid,bha )*cosϕ( s,sᵢ,A,B,C,D )  -  inferμ( s,μOH,wellbore )*(  (Fₜ*dϕ(s,sᵢ,A,B,C,D)-wᵦ(s,fluid,bha)*sin(ϕ(s,sᵢ,A,B,C,D)))^2   +    
                    ( Fₜ*sin(ϕ(s,sᵢ,A,B,C,D))*dθ(s,sᵢ,A,B,C,D) )^2  )^0.5
    SA[dFₜ]
end

function FₜJacobianSA(  u,p,t  )
    ForwardDiff.jacobian( x -> FₜSA( x,p,t ),u )
end

function solutionToVectors( sol )
    x       = sol.t
    y       = hcat(sol.u...)'
    y       = vec(   collect( y )   )
    x,y
end

function simulateFriction( runDepths::Vector{Float64},μ::Vector{Float64},
                            bha::WellConfiguration.BHA,mud::WellConfiguration.Fluid,wellbore::WellConfiguration.Wellbore,
                            ODEFunc::ODEFunction,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
    V           = Matrix{Float32}(undef,length(runDepths),length(μ) )    
    for (i,depth) in enumerate( runDepths )
        for (j,f) in enumerate( μ  )
            Fₜₒ         = SA[0.]
            newBHA      = locateBHAInHole( depth,bha )
            Sspan       = (  depth,0.0  )
            μₒ          = (  f,mud,newBHA,wellbore,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ )
            problem     =   ODEProblem( ODEFunc,
                                        Fₜₒ,
                                        Sspan,
                                        μₒ )
            sol         =   solve( problem,Rosenbrock32(  ) )   
            x,y         =   solutionToVectors( sol )
            V[i,j]      =   y[end]
        end
    end
    V
end

function torqueAndDrag( bhaPath::String,fluidPath::String,wellborePath::String,surveyPath::String,fMin::AbstractFloat = 0.05,
                        fMax::AbstractFloat = 0.60,fStep::AbstractFloat = 0.05,stepDepth::Union{Integer,AbstractFloat} = 200 )
    sᵢ,ϕᵢ,θᵢ    = readSurvey( surveyPath )
    Aᵢ,Bᵢ,Cᵢ,Dᵢ,_=calculateTrajectory( sᵢ,ϕᵢ,θᵢ )
    bha         = readBHA( bhaPath )
    mud         = readFluid( fluidPath )
    wellbore    = readWellbore( wellborePath )
    OFEFunc     = ODEFunction( FₜSA,jac=FₜJacobianSA,jac_prototype=StaticArray )
    zᵥ          = wellbore.TD
    runDepths   = collect(  2.0:stepDepth:zᵥ  )
    factors     = collect(  -fMax:fStep:-fMin  )
    RIH         = simulateFriction( runDepths,factors,bha,mud,wellbore,OFEFunc,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ );
    factors     = collect(  fMin:fStep:fMax  )
    POOH        = simulateFriction( runDepths,factors,bha,mud,wellbore,OFEFunc,sᵢ,Aᵢ,Bᵢ,Cᵢ,Dᵢ );
    (RIH, POOH, runDepths)
end

# gr()
# Plots.plot( y./1000,x,xlabel="Hook Load [klb]", ylabel="Measured Depth [ft]", 
#             title="Torque and Drag Simulation",xformatter=:plain,
#             yformatter=:plain,
#             yflip=true,xmirror=true,size=(500, 700),xlim=(-200,1000),
#             legend=false,lw=3.0,c="#d11346", minorgrid=true)  

function plotWellPathFromJulia(  )
    nothing
end

function plotResultsFromJulia(  )
    nothing
end

export torqueAndDrag, plotWellPathFromJulia, plotResultsFromJulia

end