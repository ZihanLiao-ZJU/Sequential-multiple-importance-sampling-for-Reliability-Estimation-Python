def DspRst(ite, N, Nc_Ns, N_cal):
    """Print the progress of one iteration (same layout as the MATLAB code)."""
    if ite == 1:
        print("---------------------------------------------")
        print("     i      Nsam      Nc      Ns      Ncal   ")
        print("---------------------------------------------")
    print("%6d    %6d  %6d  %6d    %6d " % (ite, N, Nc_Ns[0], Nc_Ns[1], N_cal))
