module aux

function logMeanExp( x )
  xmax = maximum(x)
  return xmax + log.(mean(exp.(x - xmax)))
end

function proximitySequence( x )
  m = length(x)
  seq = collect(1:m)
  for i = 1:m-1
    j = i + indmin(abs.(x[seq[i+1:m]] - x[seq[i]]))
    seq[[i+1,j]] = seq[[j,i+1]]
  end
  return seq[2:end]
end

function SumOfDeviationsPerBlock( y::Matrix{T}, ym::Matrix{T}, b::Integer ) where T<:AbstractFloat
  (m,n) = size(y)
  Δy = y .- ym
  B = Matrix{T}(m-b+1,n)
  B[1,:] = sum(Δy[1:b,:],1)
  for j = 1:m-b
    B[j+1,:] = B[j,:] + Δy[j+b,:] - Δy[j,:]
  end
  return B
end

function compute!( u0, P, π, f, u )
  g = (f + log.(π))'
  for i = 1:length(P)
    a = g .- u[i]
    max = maximum(a,2)
    numer = exp.(a .- max)
    denom = sum(numer,2)
    P[i] = numer ./ denom
    u0[i] = -(max + log.(denom))
  end
end

function overlapSampling( u::Vector{Matrix{Float64}} )
  m = length(u)
  f = zeros(m)
  seq = aux.proximitySequence( [mean(u[i][:,i]) for i=1:m] )
  i = 1
  for j in seq
    f[j] = f[i] + aux.logMeanExp(0.5(u[j][:,j] - u[j][:,i])) - 
                  aux.logMeanExp(0.5(u[i][:,i] - u[i][:,j]))
    i = j
  end
  return f
end

function info( msg::String, val )
  const msg_color = "\033[1;36m"
  const val_color = "\033[0;36m"
  const no_color  = "\033[0m"
  print( STDOUT, msg_color, msg )
  if (isa(val,AbstractArray))
    println( STDOUT, val_color )
    Base.showarray( STDOUT, val, false; header=false )
  elseif (val != nothing)
    print( STDOUT, val_color, val )
  end
  println( STDOUT, no_color )
end

str(a) = join( string.(a), ", " )

end
