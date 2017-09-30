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

include("auxiliary.jl")

struct State
  sample::DataFrame          # properties of sampled configurations
  potential::Function        # reduced potential function
  autocorr::Function         # autocorrelated function
  n::Int                     # number of configurations
  b::Int                     # block size
  neff::Float64              # effective sample size

  function State( sample, potential, autocorr )
    n = nrow(sample)
    b = round(Int,sqrt(n))
    y = evaluate( [autocorr], sample )
    ym = mean(y,1)
    neff = n*covarianceOBM(y,ym)[1]/covarianceOBM(y,ym,b)[1]
    isnan(neff) && error( "unable to determine effective sample size" )
    new( sample, potential, autocorr, n, b, neff )
  end
end

const MatrixVector = Vector{Matrix{Float64}}
const SymmetricMatrix = Symmetric{Float64,Matrix{Float64}}

mutable struct Case
  title::String
  verbose::Bool
  state::Vector{State}
  m::Int                     # number of states
  u::MatrixVector            # reduced energies of each configuration at all states
  P::MatrixVector            # probabilities of each configuration at all states
  π::Vector{Float64}         # mixture composition
  f::Vector{Float64}         # free energy of each state
  u0::MatrixVector           # reduced energy of each configuration at the mixture state
  O::Matrix{Float64}         # overlap matrix
  B0⁺::SymmetricMatrix
  Θ::SymmetricMatrix

  function Case(title::String; verbose::Bool = false)
    verbose && info( "Creating empty MICS case: ", title )
    new( title, verbose, Vector{State}(), 0 )
  end

  Case() = Case( "Untitled" )
end


"""
    add!( case, sample, potential )

Adds a `sample` of configurations distributed according to a given reduced `potential` to
the specified MICS `case`.

## Arguments
* `case`::`MICS.Case`: the MICS case to which the new sample will be added.
* `sample`::`DataFrames.DataFrame`: 
* `potential`::`Function`: 
* `autocorr`::`Function`:
"""
function add!( case::Case, sample::DataFrame, potential::Function,
               autocorr::Function = potential )
  case.verbose && info( "Adding new state to MICS case \"$(case.title)\"" )
  case.m == 0 || names(sample) == names(case.state[1].sample) ||
    error( "trying to add inconsistent data" )
  push!( case.state, State( sample, potential, autocorr ) )
  case.m += 1
end


"""
    covarianceOBM( y, ym, b )

Performs Overlap Batch Mean (OBM) covariance analysis with the data stored in matrix `y`,
assuming a block size `b`.
"""
function covarianceOBM( y, ym, b::Int = 1 )
  S = SumOfDeviationsPerBlock( y, ym, b )
  nmb = size(y,1) - b
  return Symmetric(syrk('U', 'T', 1.0/(b*nmb*(nmb+1)), S))
end


"""
    overlapSampling( u )

Uses the Overlap Sampling Method of Lee and Scott (1980) to compute free-energies of all
`states` relative to first one.
"""
function overlapSampling( u::MatrixVector )
  m = length(u)
  f = zeros(m)
  seq = proximitySequence( [mean(u[i][:,i]) for i=1:m] )
  i = 1
  for j in seq
    f[j] = f[i] + logMeanExp(0.5(u[j][:,j] - u[j][:,i])) - 
                  logMeanExp(0.5(u[i][:,i] - u[i][:,j]))
    i = j
  end
  return f
end


"""
    evaluate( func, sample )
"""
function evaluate( func::Vector{T}, sample::DataFrame ) where T <: Function
  n = nrow(sample)
  m = length(func)
  f = Matrix{Float64}(n,m)
  for j = 1:m
    f[:,j] = func[j]( sample )
  end
  return f
end


"""
    mics( case, X )
"""
function mics( case::Case, X::MatrixVector )
  Xm = [mean(X[i],1) for i=1:case.m]
  x0 = sum(case.π .* Xm)
  Σ0 = sum(case.π[i]^2 * covarianceOBM(X[i],Xm[i],case.state[i].b) for i=1:case.m)
  return vec(x0), Symmetric(Σ0), Xm
end


"""
    computeProbabilities!( P, π, f, u )
"""
function computeProbabilities!( P, π, f, u )
  g = (f + log.(π))'
  for i = 1:length(P)
    a = g .- u[i]
    b = exp.(a .- maximum(a,2))
    P[i] = b ./ sum(b,2)
  end
end


"""
    computeEnergiesAndProbabilities!( u0, P, π, f, u )
"""
function computeEnergiesAndProbabilities!( u0, P, π, f, u )
  g = (f + log.(π))'
  for i = 1:length(P)
    a = g .- u[i]
    max = maximum(a,2)
    numer = exp.(a .- max)
    denom = sum(numer,2)
    P[i] = numer ./ denom
    u0[i] = max + log.(denom)
  end
end


"""
    update!( case )
"""
function update( case::Case; tol::Float64 = 1.0e-8 )

  state = case.state
  verbose = case.verbose
  m = case.m
  n = [state[i].n for i=1:m]

  # Compute mixture composition:
  π = case.π = [state[i].neff for i=1:m]/sum(state[i].neff for i=1:m)
  verbose && info( "Mixture composition: ", π )

  # Compute reduced potentials:
  potentials = [state[i].potential for i=1:m]
  u = case.u = [evaluate( potentials, state[i].sample ) for i=1:m]

  # Allocate matrices:
  P = case.P = MatrixVector(m)
  u0 = case.u0 = MatrixVector(m)
  for i = 1:m
    P[i] = Matrix{Float64}(n[i],m)
    u0[i] = Matrix{Float64}(n[i],1)
  end

  # Initial guess:
  case.f = overlapSampling( u )
  verbose && info( "Initial free-energy guess:", case.f )

  # Newton-Raphson iterations:
  Δf = ones(m-1)
  iter = 0
  while any(abs.(Δf) .> tol)
    iter += 1
    computeProbabilities!( P, case.π, case.f, u )
    p0 = [mean(P[j][:,i]) for i=1:m, j=1:m]*π
    B0 = Symmetric(diagm(p0) - sum(syrk('U', 'T', π[i]/n[i], P[i]) for i=1:m))
    Δf = B0[2:m,2:m]\(π - p0)[2:m]
    case.f[2:m] += Δf
  end
  verbose && info( "Free energies after $(iter) iterations:", case.f )

  # Computation of probability covariance matrix:
  computeEnergiesAndProbabilities!( u0, P, case.π, case.f, u )
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
end


"""
    freeEnergies( case )
"""
function freeEnergies( case::Case )
  δf = sqrt.([case.Θ[i,i] - 2*case.Θ[i,1] + case.Θ[1,1] for i=1:case.m])
  return case.f, δf
end


#function reweight( case::Case, potential::Function, variable::Function )
#  m = case.m
#  state = case.state
#  u = 
#  y = [evaluate(state[i].sample,[potential,variable]) for i=1:m]
#  @show y
#end

end
