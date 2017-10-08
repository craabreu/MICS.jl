using MICS
using DataFrames
using CSV
using Base.Test

β = 1.6773985789

states = Vector{State}()
push!( states, State( readtable("log_1.dat",separator=' '), @prop(β*:E1), @prop(β*:E2) ) )
push!( states, State( readtable("log_2.dat",separator=' '), @prop(β*:E2) ) )
push!( states, State( readtable("log_3.dat",separator=' '), @prop(β*:E3) ) )
push!( states, State( readtable("log_4.dat",separator=' '), @prop(β*:E4) ) )

@time mixture = Mixture( states; title = "Test", verbose=true )

#@time f, δf = freeEnergies( mixture )


