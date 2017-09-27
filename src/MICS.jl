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

include("auxiliary.jl")

struct State
  sample::DataFrame
  potential::Function
  n::Int                     # number of configurations
  b::Int                     # block size

  State(sample,potential,n,b) = new(sample,potential,n,b)
end

mutable struct Case
  title::String
  verbose::Bool
  state::Vector{State}
  m::Int
  U::Vector{Matrix{Float64}}         # reduced energies at all states
  Um::Vector{Matrix{Float64}}
  neff::Vector{Float64}              # effective sample sizes

  π::Vector{Float64}
  f::Vector{Float64}

  Case() = new( "Untitled", false, Vector{State}(), 0 )

  function Case(title::String; verbose::Bool = false)
    verbose && println("Creating empty MICS case \"$title\"")
    new( title, verbose, Vector{State}(), 0 )
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
  case.verbose && println("Adding new state to case \"$(case.title)\"")
  case.m == 0 || names(sample) == names(case.state[1].sample) ||
    error("trying to add inconsistent data")
  n = size(sample,1)
  b = round(Int32,sqrt(n))
  push!( case.state, State(sample,potential,n,b) )
  case.m += 1
  case.verbose && println("  Number of configurations: ", n )
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
"""
    compute!( case )
"""
function compute( case::Case )

  state = case.state
  verbose = case.verbose
  m = case.m

  # Allocate matrices:
  neff = case.neff = Vector{Float64}(m)
  U = case.U = Vector{Matrix{Float64}}(m)
  Um = case.Um = Vector{Matrix{Float64}}(m)

  # Compute reduced potentials and effective sample sizes:
  verbose && println( "Correlation analysis with reduced potentials:" )
  for (i,iS) in enumerate(state)
    U[i] = Matrix{Float64}(iS.n,m)
    for (j,jS) in enumerate(state)
      U[i][:,j] = jS.potential( iS.sample )
    end
    Um[i] = mean(U[i],1)
    Σb = covarianceOBM( U[i], Um[i], iS.b )
    Σ1 = covarianceOBM( U[i], Um[i], 1 )
    neff[i] = iS.n*eigmax(Σ1)/eigmax(Σb)
  end
  π = case.π = neff/sum(neff)
  verbose && println( "Effective sample sizes: ", neff )
  verbose && println( "Marginal state probabilities: ", π )

  # Compute initial guess for free energy differences:
  verbose && println( "Overlap Sampling calculations:" )
  f = case.f = overlapSampling( case )
  verbose && println( "Free-energy initial guess: ", f )

  # Newton-Raphson iterations:
  verbose && println("Newton-Raphson iterations:")


#  Δf = ones(m-1)
#  case.M = Matrix{Float64}(m,m)
#  B = Matrix{Float64}(m,m)
#  while any(abs.(Δf) .> tol)
#    for j = 1:m
#      for k = 1:n[j]
#        state[j].P[k,:] = posteriors(case.π, state[j].U[k,:], case.f)
#      end
#      case.M[:,j] = [mean(state[j].P[:,i]) for i = 1:m]
#      state[j].Ω = Symmetric(syrk('U', 'T', 1.0, state[j].P)/n[j])
#    end
#    Mπ = case.M*case.π
#    B = Diagonal(Mπ) - sum(case.π[i]*state[i].Ω for i = 1:m)
#    s = case.π - Mπ
#    Δf = B[2:m,2:m]\s[2:m]
#    case.f[2:m] += Δf
#    case.verbose && println("> f = ", case.f)
#  end

end

#-------------------------------------------------------------------------------------------
end
