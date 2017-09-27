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

struct State
  sample::DataFrame          # properties of sampled configurations
  potential::Function        # reduced potential function
  n::Int                     # number of configurations
  b::Int                     # block size
end

mutable struct Case
  title::String
  verbose::Bool
  state::Vector{State}
  upToDate::Bool                     # true if energies have been computed
  m::Int                             # number of states
  U::Vector{Matrix{Float64}}         # reduced energies
  Um::Vector{Matrix{Float64}}        # sample means of reduced energies
  neff::Vector{Float64}              # effective sample sizes
  π::Vector{Float64}                 # marginal probabilities
  f::Vector{Float64}                 # free energies
  B0⁺::SymmetricMatrix
  Θ::SymmetricMatrix

  Case() = new( "Untitled", false, Vector{State}(), 0, false )

  function Case(title::String; verbose::Bool = false)
    verbose && info( prefix="Creating empty MICS case: ", title )
    new( title, verbose, Vector{State}(), 0, false )
  end
end

#-------------------------------------------------------------------------------------------
function show( io::IO, case::Case )
  state = case.state
  if case.m > 0
    s = case.m > 1 ? "s" : ""
    println(io, "MICS case \"$(case.title)\" contains $(case.m) state$(s):")
    println(io, "  - Sample size$(s): ",[size(state[i].sample,1) for i=1:case.m])
    s = size(state[1].sample,2) > 1 ? "ies" : "y"
    println(io, "  - Propert$(s): ", join(map(string,names(state[1].sample)),", "))
  else
    println(io, "MICS case \"$(case.title)\" contains no states.")
  end
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
    info( prefix="Adding new state to MICS case $(case.title ): ", n, " configurations" )
  case.m == 0 || names(sample) == names(case.state[1].sample) ||
    error( "trying to add inconsistent data" )
  b = round(Int32,sqrt(n))
  push!( case.state, State(sample,potential,n,b) )
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
  Symmetric(syrk('U', 'T', 1.0/(b*n*(n+1)), S))    # Covariance matrix
end

#-------------------------------------------------------------------------------------------
"""
    overlapSampling( states )

Uses the Overlap Sampling Method of Lee and Scott (1980) to compute free-energies of all
`states` relative to first one.
"""
function overlapSampling( case )
  f = zeros(case.m)
  seq = proximitySequence( [case.Um[i][i] for i=1:case.m] )
  i = 1
  for j in seq[2:end]
    f[j] = f[i] + logMeanExp(0.5(case.U[j][:,j] - case.U[j][:,i])) -
                  logMeanExp(0.5(case.U[i][:,i] - case.U[i][:,j]))
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
function posteriors( π, f, u )
  a = f - u
  p = π.*exp.(a - maximum(a))
  return p/sum(p)
end

#-------------------------------------------------------------------------------------------
function mics( case::Case, X::Vector{Matrix{Float64}} )
  Xm = [mean(X[i],1) for i=1:case.m]
  x0 = vec(sum(case.π.*Xm))
  Σ0 = sum(case.π[i]^2*covarianceOBM(X[i],Xm[i],case.state[i].b) for i=1:case.m)
  return vec(x0), Symmetric(Σ0)
end

#-------------------------------------------------------------------------------------------
"""
    compute!( case )
"""
function compute( case::Case; tol::Float64 = 1.0e-8 )

  state = case.state
  verbose = case.verbose
  m = case.m
  n = [state[i].n for i=1:m]

  # Evaluate reduced potentials at all states and their sample means:
  U = case.U = evaluate( case, [state[i].potential for i=1:m] )
  Um = case.Um = [mean(U[i],1) for i=1:m]

  # Compute effective sample sizes and marginal state probabilities:
  Σb = [covarianceOBM(U[i],Um[i],state[i].b) for i=1:m]
  Σ1 = [covarianceOBM(U[i],Um[i],1) for i=1:m]
  neff = case.neff = [n[i]*eigmax(Σ1[i])/eigmax(Σb[i]) for i=1:m]
  π = case.π = neff/sum(neff)

  verbose && (info( prefix="Effective sample sizes: ", neff );
              info( prefix="Marginal state probabilities: ", π ))

  # Newton-Raphson iterations:
  δf = ones(m-1)
  iter = 0
  f = case.f = overlapSampling( case )
  verbose && info( prefix="Initial free-energy guess: ", f )
  while any(abs.(δf) .> tol)
    iter += 1
    P = [mapslices(u->posteriors(π,f,u),U[i],2) for i=1:m]
    p0 = vec(sum(π[i]*mean(P[i],1) for i=1:m))
    mB0 = Symmetric(sum(syrk('U', 'T', π[i]/n[i], P[i]) for i=1:m) - diagm(p0))
    g = p0 - π
    δf = mB0[2:m,2:m]\g[2:m]
    f[2:m] += δf
  end
  verbose && info( prefix="Free energies after $(iter) iterations: ", f )

  # Computation of probability covariance matrix:
  P = [mapslices(u->posteriors(π,f,u),U[i],2) for i=1:m]
  p0, Σ0 = mics( case, P )
  B0 = Symmetric(diagm(p0) - sum(Symmetric(syrk('U', 'T', π[i]/n[i], P[i])) for i=1:m))

  # Computation of free-energy covariance matrix:
  D, V = eig(B0)
  D⁺ = diagm(map(x -> x > tol*maximum(D)? 1.0/x : 0.0, D))
  B0⁺ = case.B0⁺ = Symmetric(V*D⁺*V')
  case.Θ = Symmetric(B0⁺*Σ0*B0⁺)

  case.upToDate = true
end

#-------------------------------------------------------------------------------------------
end
