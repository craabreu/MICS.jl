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

mutable struct State
  sample::DataFrame
  potential::Function
  n::Int32                   # number of configurations
  b::Int32                   # block size
  neff::Float64              # effective sample size
  U::Matrix{Float64}         # reduced energies at all states
  Um::Matrix{Float64}        # sample means of reduced energies

  State(sample,potential,n,b) = new(sample,potential,n,b)
end

struct Case
  title::String
  verbose::Bool
  state::Vector{State}

  Case() = new( "Untitled", false, Vector{State}() )

  function Case(title::String; verbose::Bool = false)
    verbose && println("Creating empty MICS case \"$title\"")
    new( title, verbose, Vector{State}() )
  end
end

#-------------------------------------------------------------------------------------------
function show( io::IO, case::Case )
  state = case.state
  n = length(state)
  if n > 0
    s = n > 1 ? "s" : ""
    println(io, "MICS case \"$(case.title)\" contains $n state$(s):")
    println(io, "  - Sample size$(s): ",[size(state[i].sample,1) for i=1:n])
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
  length(case.state) == 0 || names(sample) == names(case.state[1].sample) ||
    error("trying to add inconsistent data")
  n = size(sample,1)
  b = round(Int32,sqrt(n))
  push!( case.state, State(sample,potential,n,b) )
  case.verbose && println("  Number of configurations: ", n )
end # function add!

#-------------------------------------------------------------------------------------------
"""
    SumOfDeviationsPerBlock( y, ym, b )
"""
function SumOfDeviationsPerBlock( y, ym, b )
  (m,n) = size(y)                                  # m = sample size, n = # of properties
  Δy = broadcast(+,y,-ym)                          # Deviations from the sample means
  B = Matrix{Float64}(m-b+1,n)
  B[1,:] = sum(Δy[1:b,:],1)                        # Sum of deviations of first block
  for j = 1:m-b
    B[j+1,:] = B[j,:] + Δy[j+b,:] - Δy[j,:]        # Sum of deviations of (j+1)-th block
  end
  return B
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
    compute!( case )
"""
function compute( case::Case )

  state = case.state
  m = length(state)

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
end # compute

#-------------------------------------------------------------------------------------------
end # module
