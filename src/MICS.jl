"""
    module MICS

MICS (Mixture of Independently Collected Samples) is a method for performing free-energy
computation and property reweighting using samples collected at multiple states. Its main
distinguishing feature is the possibility of employing all data from autocorrelated samples,
without sub-sampling.
"""
module MICS

using DataFrames
using Base.LinAlg.BLAS

export State,
       Mixture,
       covariance

include("aux.jl")

"""
    struct State

An equilibrium state aimed to enroll in a MICS analysis
"""
struct State
  sample::DataFrame          # properties of sampled configurations
  potential::Function        # reduced potential function
  autocorr::Function         # autocorrelated function
  n::Int                     # number of configurations
  b::Int                     # block size
  neff::Int                  # effective sample size
end

"""
# Constructor

    State( sample, potential[, autocorr] )

# Arguments

* `sample::DataFrames.DataFrame`: a data frame whose rows represent to configurations
                                  sampled according to a given probability distribution and
                                  whose columns contain a number of properties evaluated for
                                  such configurations.
* `potential::Function`: the reduced potential that defines the equilibrium state. It must
                         be a function that receives `x::DataFrame` and returns `u::T`,
                         where `T<:AbstractArray{Float64,1}` and `length(u) == nrow(x)`.
* `autocorr::Function=potential`: a function similar to `potential`, but returning some
                                  autocorrelated property to be used for determining the
                                  effective sample size.
"""
function State( sample, potential, autocorr=potential )
  n = nrow(sample)
  b = round(Int,sqrt(n))
  y = evaluate( [autocorr], sample )
  neff = n*covariance(y,1)[1]/covariance(y,b)[1]
  isfinite(neff) || error( "unable to determine effective sample size" )
  State( sample, potential, autocorr, n, b, round(Int,neff) )
end

"""
    struct Mixture

A mixture of independently collected samples (MICS)
"""
struct Mixture
  title::String
  state::Vector{State}
  names::Vector{Symbol}
  m::Int                       # number of states
  n::Vector{Int}               # sample size at each state
  u::Vector{Matrix{Float64}}   # reduced energies of each configuration at all states
  π::Vector{Float64}           # mixture composition
  f::Vector{Float64}           # free energy of each state
  P::Vector{Matrix{Float64}}   # probabilities of each configuration at all states
  u0::Vector{Matrix{Float64}}  # reduced energy of each configuration at the mixture state
  B0⁺::Matrix{Float64}
  Θ::Matrix{Float64}
end

"""
# Constructor

    Mixture( states; <keyword arguments> )

# Arguments

* `states::MICS.Vector{State}`: 
* `title::String=\"Untitled\"`: 
* `verbose::Bool=false`:
* `tol::Float64=1.0E-8`:
"""
function Mixture( states::Vector{State}; title::String = "Untitled",
                  verbose::Bool = false, tol::Float64 = 1.0E-8 )

  verbose && aux.info( "Setting up MICS case: ", title )

  m = length(states)
  verbose && aux.info( "Number of states: ", m )
  m == 0 && error( "state set is empty" )

  properties = names(states[1].sample)
  verbose && aux.info( "Properties: ", aux.str(properties) )
  all([all(names(states[i].sample) .== properties) for i=1:m]) ||
    error( "inconsistent data" )

  n = [nrow(states[i].sample) for i=1:m]
  verbose && aux.info( "Sample sizes: ", aux.str(n) )

  neff = [states[i].neff for i=1:m]
  verbose && aux.info( "Effective sample sizes: ", aux.str(neff) )

  π = neff/sum(neff)
  verbose && aux.info( "Mixture composition: ", π )

  potentials = [states[i].potential for i=1:m]
  u = [evaluate(potentials, states[i].sample) for i=1:m]

  P = Vector{Matrix{Float64}}(m)
  u0 = Vector{Matrix{Float64}}(m)
  for i = 1:m
    P[i] = Matrix{Float64}(n[i],m)
    u0[i] = Matrix{Float64}(n[i],1)
  end

  f = aux.overlapSampling( u )
  verbose && aux.info( "Initial free-energy guess:", f )

  iter = 1
  aux.compute!( u0, P, π, f, u )
  p0 = [mean(P[j][:,i]) for i=1:m, j=1:m]*π
  B0 = Symmetric(diagm(p0) - sum(syrk('U', 'T', π[i]/n[i], P[i]) for i=1:m))
  Δf = B0[2:m,2:m]\(π - p0)[2:m]
  while any(abs.(Δf) .> tol)
    iter += 1
    f[2:m] += Δf
    aux.compute!( u0, P, π, f, u )
    p0 = [mean(P[j][:,i]) for i=1:m, j=1:m]*π
    B0 = Symmetric(diagm(p0) - sum(syrk('U', 'T', π[i]/n[i], P[i]) for i=1:m))
    Δf = B0[2:m,2:m]\(π - p0)[2:m]
  end
  verbose && aux.info( "Free energies after $(iter) iterations:", f )

  (D, V) = eig(B0)
  D⁺ = diagm(map(x-> abs(x) < tol ? 0.0 : 1.0/x, D))
  B0⁺ = Symmetric(V*D⁺*V')
  Σ0 = sum(π[i]^2*covariance(P[i],states[i].b) for i=1:m)
  Θ = Symmetric(B0⁺*Σ0*B0⁺)
  verbose && aux.info( "Free-energy covariance matrix:", full(Θ) )

  Mixture( title, states, properties, m, n, u, π, f, P, u0, B0⁺, Θ )
end

"""
    covariance( y, b )

Performs Overlap Batch Mean (OBM) covariance analysis with the data stored in matrix `y`,
assuming a block size `b`.
"""
function covariance( y::Matrix{T}, b::Integer ) where T<:AbstractFloat
  S = aux.SumOfDeviationsPerBlock( y, b )
  nmb = size(y,1) - b
  return Symmetric(syrk('U', 'T', 1.0/(b*nmb*(nmb+1)), S))
end

"""
    crossCovariance( y, z, b )

Performs Overlap Batch Mean (OBM) cross-covariance analysis with the data stored in matrices
`y` and `z`, assuming a block size `b`.
"""
function covariance( y::Matrix{T}, z::Matrix{T}, b::Integer ) where T<:AbstractFloat
  Sy = SumOfDeviationsPerBlock( y, b )
  Sz = SumOfDeviationsPerBlock( z, b )
  nmb = size(y,1) - b
  return gemm('T', 'N', 1.0/(b*nmb*(nmb+1)), Sy, Sz)
end

"""
    evaluate( functions, sample )
"""
function evaluate( functions::Vector{T}, sample::DataFrame ) where T <: Function
  m = length(functions)
  f = Matrix{Float64}(nrow(sample),m)
  for j = 1:m
    f[:,j] = functions[j]( sample )
  end
  return f
end

"""
    freeEnergies( mixture )
"""
function freeEnergies( mixture::Mixture )
  δf = sqrt.([mixture.Θ[i,i] - 2*mixture.Θ[i,1] + mixture.Θ[1,1] for i=1:case.m])
  return mixture.f, δf
end


#function reweight( case::Case, potential::Function, variable::Function )
#  m = case.m
#  state = case.state
#  u = 
#  y = [evaluate(state[i].sample,[potential,variable]) for i=1:m]
#  @show y
#end

end
