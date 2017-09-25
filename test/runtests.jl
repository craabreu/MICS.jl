using MICS
using DataFrames
using CSV
using Base.Test

case = MICS.Case( "Test"; verbose=true )
MICS.add!( case, readtable("log_1.dat",separator=' '), X->X(:E1) )
MICS.add!( case, readtable("log_2.dat",separator=' '), X->X(:E2) )
MICS.add!( case, readtable("log_3.dat",separator=' '), X->X(:E3) )
MICS.add!( case, readtable("log_4.dat",separator=' '), X->X(:E4) )

println( case )
#@test 1 == 2
