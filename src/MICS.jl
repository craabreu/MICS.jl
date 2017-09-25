"""
    module MICS

MICS (Mixture of Independently Collected Samples) is a method for performing free-energy
computation and property reweighting using samples collected at multiple states. Its main
distinguishing feature is the possibility of employing all data from autocorrelated samples,
without sub-sampling.
"""
module MICS

using DataFrames

import Base.show

struct State
  sample::DataFrame
  potential::Function
end # struct State

struct Case
  title::String
  verbose::Bool
  state::Vector{State}

  Case() = new( "Untitled", false, Vector{State}() )

  function Case(title::String; verbose::Bool = false)
    verbose && println("Creating MICS-analysis case \"$title\"")
    new( title, verbose, Vector{State}() )
  end
end # struct Case

#-------------------------------------------------------------------------------------------
function show( io::IO, case::Case )
  n = length(case.state)
  if n > 0
    s = n > 1 ? "s" : ""
    println(io, "MICS case \"$(case.title)\" contains $n state$(s):")
    println(io, "  - Sample size$(s): ",[size(case.state[i].sample,1) for i=1:n])
    s = size(case.state[1].sample,2) > 1 ? "ies" : "y"
    println(io, "  - Propert$(s): ", join(map(string,names(case.state[1].sample)),", "))
  else
    println(io, "MICS case \"$(case.title)\" contains no states.")
  end
end

#-------------------------------------------------------------------------------------------
"""
    add!( case, sample, potential )

Adds a `sample` of configurations distributed according to a given reduced `potential` to
the specified MICS-analysis `case`.

## Arguments
* `case::MICS.Case`: 
* `sample::DataFrames.AbstractDataFrame`: 
* `potential::Function`: 
"""
function add!( case::Case, sample::AbstractDataFrame, potential::Function )
  case.verbose && println("Adding new state to case \"$(case.title)\"")
  length(case.state) == 0 || names(sample) == names(case.state[1].sample) ||
    error("trying to add inconsistent data")
  push!( case.state, State( sample, potential ) )
  case.verbose && println("  Number of configurations: ", size(sample,1))
end # function add!

#-------------------------------------------------------------------------------------------

end # module
