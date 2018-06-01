This is an implementation of the *Obtenebrator*, which is an algorithm that takes samples from a mixture of product distributions and tries to reconstruct the parameters of the distributions.

For each coordinate, it will give you a list of approximate parameters, and it should work in polynomial time and with a polynomial number of samples (although I am not sure what polynomial). I should do an analysis of it, but the main problem is that it doesn't really work unless the means of the individual product distributions are generic.

See explanation-doesntwork.pdf for an explanation of this version of the algorithm. (It "doesn't work" in the sense that it doesn't deal with every case; it does work in the sense that the program runs and produces correct output.)
