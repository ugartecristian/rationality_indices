# rationality_indices

Implementation of Several Indices of Economic Rationality. 

Each method takes as input two N x K matrices, where N is the number of observations, and K is the number of goods:<br>

	- A matrix *p* of prices, where p[i, j] is the price of the k<sup>th</sup> good in the i<sup>th</sup> observation. Prices must be normalized such that total expenditure in each observation equals one.
	
	- A matrix *x* of consumption bundles, where x[i, j] is the consumption of the k<sup>th</sup> good in the i<sup>th</sup> observation.

- *CCEI.py* solves the Critical Cost Efficiency Index (Afriat, 1973).

- *HM.py* solves the Houtman and Masks (1985) Index. 
	- It offers three different options to find the solutions: 
		- solving the related Minimum Feedback Vertex Set Problem via Linear Integer Programming
		- using the method proposed by Heufer and Hjertstrand (2015), and 
		- using the method proposed by Demuynck and Rehbeck (2023).

- *Varian.py* solves the Varian (1990) Index.
	- It uses the method proposed by Demuynck and Rehbeck (2023).

- *MM.py* solves the Minimum Mistakes Index proposed in Ugarte (2023). 
	- It proceeds by solving the related Minimum Feedback Arc Set Problem, using the methodology in Baharev et. al. (2021) and the implementation of this method by Baharev (https://github.com/baharev/sdopt-tearing)
		- The implementation proposed by Baharev is written in Python 2. I converted to Python 3 using the 2to3 program (https://docs.python.org/3.9/library/2to3.html)