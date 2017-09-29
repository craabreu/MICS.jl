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

function info( io::IO, msg::String, val )
  const msg_color = "\033[1;36m"
  const val_color = "\033[0;36m"
  const no_color  = "\033[0m"
  print( msg_color, msg )
  if (isa(val,AbstractArray))
    println( val_color )
    Base.showarray( io, val, false; header=false )
  elseif (val != nothing)
    print( io, val_color, val )
  end
  println( no_color )
end

info(msg::String,val) = info(STDOUT,msg,val)
info(msg::String) = info(msg,nothing)
