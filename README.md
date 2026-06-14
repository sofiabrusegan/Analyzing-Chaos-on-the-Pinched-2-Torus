# Analyzing Chaos on the pinched 2-torus

This project contains all of the code used for figures, diagrams and computations used in my Bachelor thesis. The thesis focuses on detecting any chaotic behaviour that arises from adding a perturbation to a Hamiltonian system using the Melnikov method. The thesis explores a specific case where the family of homoclinic orbits form a pinched 2-torus, hence Melnikov's method has been adapted. The stable and unstable manifolds no longer align once the perturbation is added, hence the Melnikov method is used to detect their transverse intersections, which in turn imply the existence of a Smale horseshoe.

## Contents of the project:
- A Python module containing all functions used throughout the thesis. This includes, but is not limited to, the Hamiltonian, the equations of motions, an integrating function, and a function to create Poincaré maps with Hénon's method.
- A Jupyter notebook containing the code used to create the figures

 ## Potential uses of the code:
 - Creating Poincaré maps with Hénon's method for four-dimensional Hamiltonian systems
 - Conversion between Cartesian and polar coordinates
 - Integrating equations of motions with respect to Cartesian or polar coordinates

## Further reading and contacts
The thesis which this code supports can be found at: [insert link]

For questions or comments, please contact: sofia.brusegan@gmail.com

This code is maintained by Sofia Brusegan.
