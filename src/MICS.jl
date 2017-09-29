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

import Base.show
import Base.mean

include("auxiliary.jl")

const SymmetricMatrix = Symmetric{Float64,Matrix{Float64}}

mutable struct State
  sample::DataFrame          # properties of sampled configurations
  potential::Function        # reduced potential function
  n::Int                     # number of configurations
  b::Int                     # block size
  State(sample,potential,n) = new(sample,potential,n,round(Int,sqrt(n)))
end

mutable struct Case
  title::String
  verbose::Bool
  state::Vector{State}
  upToDate::Bool                     # true if energies have been computed
  m::Int                             # number of states
  u::Vector{Matrix{Float64}}
  π::Vector{Float64}                 # marginal probabilities
  f::Vector{Float64}                 # free energies
  δ²f::Vector{Float64}               # free-energy square errors
  O::Matrix{Float64}                 # overlap matrix
  B0⁺::SymmetricMatrix
  Θ::SymmetricMatrix

  Case() = new( "Untitled", false, Vector{State}(), 0, false )

  function Case(title::String; verbose::Bool = false)
    verbose && info( "Creating empty MICS case: ", title )
    new( title, verbose, Vector{State}(), 0, false )
  end
end

#-------------------------------------------------------------------------------------------
function info( msg::String, x )
  println( "\033[1;36m", msg, "\033[0;36m")
  if (isa(x,AbstractArray))
    Base.showarray( STDOUT, x, false; header=false )
  else
    print( STDOUT, " ", x )
  end
  println("\033[0m")
end

#-------------------------------------------------------------------------------------------
"""
    add!( case, sample, potential )

Adds a `sample` of configurations distributed according to a given reduced `potential` to
the specified MICS `case`.

## Arguments
* `case::MICS.Case`: 
* `sample::DataFrames.DataFrame`: 
* `potential::Function`: 
"""
function add!( case::Case, sample::DataFrame, potential::Function )
  n = size(sample,1)
  case.verbose &&
    info( "Adding new state to MICS case $(case.title ): ", "$(n) configurations" )
  case.m == 0 || names(sample) == names(case.state[1].sample) ||
    error( "trying to add inconsistent data" )
#  b = round(Int32,sqrt(n))
  push!( case.state, State(sample,potential,n) )
  case.m += 1
  case.upToDate = false
end

#-------------------------------------------------------------------------------------------
"""
    covarianceOBM( y, ym, b )

Performs Overlap Batch Mean (OBM) covariance analysis with the data stored in matrix `y`,
assuming a block size `b`.
"""
function covarianceOBM( y, ym, b )
  S = SumOfDeviationsPerBlock( y, ym, b )          # Blockwise sum of deviations
  n = size(y,1) - b                                # Number of blocks minus 1
  return Symmetric(syrk('U', 'T', 1.0/(b*n*(n+1)), S))    # Covariance matrix
end

function OBM( y, b )
  ym = mean(ym,2)
  S = SumOfDeviationsPerBlock( y, ym, b )
  n = size(y,1)
  Σ = syrk('U', 'T', 1.0/(b*(n-b)*(n-b+1)), S)
  return vec(ym), Symmetric(Σ)
end

#-------------------------------------------------------------------------------------------
"""
    overlapSampling( states )

Uses the Overlap Sampling Method of Lee and Scott (1980) to compute free-energies of all
`states` relative to first one.
"""
function overlapSampling( case )
  f = zeros(case.m)
  u = case.u
  seq = proximitySequence( [mean(u[i][:,i]) for i=1:case.m] )
  i = 1
  for j in seq[2:end]
    f[j] = f[i] + logMeanExp(0.5(u[j][:,j] - u[j][:,i])) - 
                  logMeanExp(0.5(u[i][:,i] - u[i][:,j]))
    i = j
  end
  return f
end

#-------------------------------------------------------------------------------------------
function evaluate( case::Case, func::Vector{Function} )
  m = case.m
  n = length(func)
  f = Vector{Matrix{Float64}}(m)
  for (i,S) in enumerate(case.state)
    f[i] = Matrix{Float64}(S.n,n)
    for j = 1:n
      f[i][:,j] = func[j]( S.sample )
    end
  end
  return f
end

#-------------------------------------------------------------------------------------------
function mics( case::Case, X::Vector{Matrix{Float64}} )
  Xm = [mean(X[i],1) for i=1:case.m]
  x0 = sum(case.π .* Xm)
  Σ0 = sum(case.π[i]^2 * covarianceOBM(X[i],Xm[i],case.state[i].b) for i=1:case.m)
  return vec(x0), Symmetric(Σ0), Xm
end

#-------------------------------------------------------------------------------------------
function compute_probabilities!( P, π, f, u )
  g = (f + log.(π))'
  for i = 1:length(P)
    a = g .- u[i]
    b = exp.(a .- maximum(a,2))
    P[i] = b ./ sum(b,2)
  end
end

#-------------------------------------------------------------------------------------------
"""
    compute!( case )
"""
function compute( case::Case; tol::Float64 = 1.0e-8, priors::Vector{Float64} = [] )

  state = case.state
  verbose = case.verbose
  m = case.m
  n = [state[i].n for i=1:m]

  # Allocate matrices and compute the reduced potentials:
  u = case.u = Vector{Matrix{Float64}}(m)
  P = Vector{Matrix{Float64}}(m)
  for i = 1:m
    u[i] = Matrix{Float64}(n[i],m)
    P[i] = Matrix{Float64}(n[i],m)
    for j = 1:m
      u[i][:,j] = state[j].potential( state[i].sample )
    end
  end

  if isempty(priors)

    # Compute effective sample sizes:
    Σb = [covarianceOBM(u[i],Um[i],state[i].b) for i=1:m]
    Σ1 = [covarianceOBM(u[i],Um[i],1) for i=1:m]
    neff = [n[i]*eigmax(Σ1[i])/eigmax(Σb[i]) for i=1:m]
    verbose && info( "Effective sample sizes: ", neff )

    # Compute marginal state probabilities:
    π = case.π = neff/sum(neff)

  else

    length(priors) == m || error( "wrong number of specified priors" )
    π = case.π = priors/sum(priors)

  end
  verbose && info( "Marginal state probabilities: ", π )

  # Newton-Raphson iterations:
  f = case.f = overlapSampling( case )
  verbose && info( "Initial free-energy guess:", f )

  Δf = ones(m-1)
  iter = 0
  while any(abs.(Δf) .> tol)
    iter += 1
    compute_probabilities!( P, case.π, case.f, u )
    p0 = [mean(P[j][:,i]) for i=1:m, j=1:m]*π
    B0 = Symmetric(diagm(p0) - sum(syrk('U', 'T', π[i]/n[i], P[i]) for i=1:m))
    Δf = B0[2:m,2:m]\(π - p0)[2:m]
    f[2:m] += Δf
  end
  verbose && info( "Free energies after $(iter) iterations:", f )

  # Computation of probability covariance matrix:
  compute_probabilities!( P, case.π, case.f, u )
  p0, Σ0, pm = mics( case, P )
  B0 = Symmetric(diagm(p0) - sum(syrk('U', 'T', π[i]/n[i], P[i]) for i=1:m))

  # Compute overlap matrix:
  case.O = [pm[j][i] for i=1:m, j=1:m]
  verbose && info( "Overlap matrix:", case.O )

  # Computation of free-energy covariance matrix:
  (D, V) = eig(B0)
  D⁺ = diagm(map(x-> abs(x) < tol ? 0.0 : 1.0/x, D))
  case.B0⁺ = Symmetric(V*D⁺*V')
  case.Θ = Symmetric(case.B0⁺*Σ0*case.B0⁺)
  verbose && info( "Free-energy covariance matrix:", full(case.Θ) )

  # Computation of free-energy uncertainties:
  case.δ²f = [case.Θ[i,i] - 2*case.Θ[i,1] + case.Θ[1,1] for i=1:m]
  verbose && info( "Free-energy uncertainties:", sqrt.(case.δ²f) )

  case.upToDate = true
end

#-------------------------------------------------------------------------------------------
end
