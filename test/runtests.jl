using MICS
using DataFrames
using CSV
using Base.Test

β = 1.6773985789

states = Vector{State}()
push!( states, State( readtable("log_1.dat",separator=' '), x->β*x[:E1], x->β*x[:E2]) )
push!( states, State( readtable("log_2.dat",separator=' '), x->β*x[:E2]) )
push!( states, State( readtable("log_3.dat",separator=' '), x->β*x[:E3]) )
push!( states, State( readtable("log_4.dat",separator=' '), x->β*x[:E4]) )

@time mixture = Mixture( states; title = "Test", verbose=true )

@time f, δf = freeEnergies( mixture )


