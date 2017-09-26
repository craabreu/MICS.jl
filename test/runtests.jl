using MICS
using DataFrames
using CSV
using Base.Test

case = MICS.Case( "Test"; verbose=true )
β = 1.6773985789
MICS.add!( case, readtable("log_1.dat",separator=' '), x->β*x[:E1] )
MICS.add!( case, readtable("log_2.dat",separator=' '), x->β*x[:E2] )
MICS.add!( case, readtable("log_3.dat",separator=' '), x->β*x[:E3] )
MICS.add!( case, readtable("log_4.dat",separator=' '), x->β*x[:E4] )

MICS.compute( case )

println( case )
#@test 1 == 2
