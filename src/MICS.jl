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

mutable struct State
  sample::DataFrame
  potential::Function
  n::Int                     # number of configurations
  b::Int                     # block size
  neff::Float64              # effective sample size
  U::Matrix{Float64}         # reduced energies at all states
  Um::Matrix{Float64}        # sample means of reduced energies

  State(sample,potential,n,b) = new(sample,potential,n,b)
end

mutable struct Case
  title::String
  verbose::Bool
  state::Vector{State}
  m::Int32

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
    covarianceOBM( y; b )

Performs an Overlap Batch Mean analysis with the data stored in matrix `y` and `z`,
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
function overlapSampling( states )
  m = length(states)
  f = zeros(m)
  seq = proximitySequence( [states[i].Um[i] for i=1:m] )
  i = 1
  for j in seq[2:end]
    f[j] = f[i] + logMeanExp(0.5(states[j].U[:,j] - states[j].U[:,i])) -
                  logMeanExp(0.5(states[i].U[:,i] - states[i].U[:,j]))
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

  # Compute reduced potentials and effective sample sizes:
  for i in state
    i.U = Matrix{Float64}(i.n,m)
    for j = 1:m
      i.U[:,j] = state[j].potential( i.sample )
    end
    i.Um = mean(i.U,1)
    Σb = covarianceOBM( i.U, i.Um, i.b )
    Σ1 = covarianceOBM( i.U, i.Um, 1 )
    i.neff = i.n*eigmax(Σ1)/eigmax(Σb)
  end

  # Compute initial guess for free energy differences:
  verbose && println( "Computing initial guess via Overlap Sampling" )
  f = overlapSampling( state )
  verbose && @show f

end

#-------------------------------------------------------------------------------------------
end
