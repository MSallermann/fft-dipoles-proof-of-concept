# fft-dipoles-proof-of-concept
Proof-of-concept implementation of an FFT-algorithm
to calculate gradients due to magnetic dipole-dipole interactions
on a bravais lattice with basis and open boundary conditions.

Only tested on python 3.6.5.


DISCLAIMER: This is not intended to be a reference implementation and 
            it is likely to be very slow. It doesnt try to optimize
            the implementation in any way.
            The only purpose of this code is to show that the basic
            math and logic of the algorithm works correctly.

-------------------------------------------------------------------------
    USE
-------------------------------------------------------------------------

    Modify the beginning of proof_full.py to setup a geometry or use
    standard values

    Then:
        > python proof_full.py

    Output is in Output/output_full.txt by default.
    Scroll to the end of outputfile to find comparison between Brute Force
    and this algorithm.

    Dont bother with the special cases folder, the code in there will 
    likely not work anymore.

-------------------------------------------------------------------------
   BRIEF DESCRIPTION OF ALGORITHM
-------------------------------------------------------------------------

   Let B be the number of atoms per basis cell

   Steps:
   1.  Separate the magnetic moments in B sets so that each sets only 
       contains magnetic moments that live on the same sublattice
   2.  Calculate equally as many sets of dipole-matrices where each set
       describes interactions between two specific sublattices
   3.  Pad the sets of dipole matrices and magnetic moments, so that the
       convolution of the padded sets is equal to the correct physical
       calculation (with respect to the boundary condition)
   4.  Perform the convolutions via the convolution theorem and add them
------------------------------------------------------------------------