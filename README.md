# This is the companion repository for the Spheroid Publication

This is the template for a spheroid_publication analysis.

There will be 4 different analyses, implemented as 4 different branches:  
1) A main paper analysis ('master'): orchestrates all analyses  
2) Steady state analysis ('steady'): 4 different cell lines  
3) Overexpression analysis ('oexp'): x constructs overexpressed in 360 spheres  
4) Brightfield analysis ('bf'): Analysis of the brightfield images  

There is a dependency such that analyses with lower number depend on the other analyses.

The analyses are orchestrated by the main paper analysis, which contains the other analyses as submodules.

