# Developing a Coupled Matrix-Tensor Factorization Model in Python to Analyze Mobility Data for Disaster Response
2022-23 Synopsys Research Project

Project Number 124-H10-71

Lisa Fung • Saratoga High School (Competed under Los Gatos High School for Advanced Science Research class)

Mentored by Evgeny Noi, Graduate Student Researcher at UC Santa Barbara Department of Geography

### Poster

![Poster_CMTF_mobility_analysis_Lisa_Fung](https://user-images.githubusercontent.com/71937811/229308719-6ca408b7-cb12-4bbf-bf52-4ea2ffd52332.png)
Poster pdf: [Poster_CMTF_mobility_analysis_Lisa_Fung.pdf](https://github.com/lfun1/cmtf-mobility-analysis/files/11130308/Poster_CMTF_mobility_analysis_Lisa_Fung.pdf)

### Abstract

Wildfires have repeatedly threatened cities in California. To improve evacuation routes and disaster response, we must understand the underlying factors of human mobility. However, the hidden factors that account for human movement, such as rising temperatures, may not be obvious from initial inspection of the data. Moreover, it is difficult to combine different mobility datasets into a single dataset and find their shared hidden factors.

Addressing this issue, Coupled Matrix-Tensor Factorization (CMTF) can combine and simultaneously decompose large sets of real-world data using tensors and additional matrices. However, most CMTF implementations are based on Alternating Least Squares (ALS) optimization, which performs poorly on real-world datasets with missing values.

I developed and programmed a CMTF algorithm using nonlinear conjugate gradient optimization to improve the model's accuracy. My gradient-based CMTF performed with RMSE under 0.57 when decomposing a 50x50x50 tensor and 50x50 matrix into at least 2 components, which was on par with the existing TensorLy CMTF-ALS function.

Afterward, I used the Core Consistency Diagnostic and Factor Match Score to determine an optimal CMTF model for studying mobility in the Greater Los Angeles area. Using geographic representations and time series plots, I uncovered, visualized, and interpreted the hidden factors affecting human movement: Natural Parks and Recreation, Shopping Areas, and Housing Areas.

In real-world applications, cities can collect mobility data through different sources and utilize my CMTF model to uncover hidden factors in human mobility. This can be used to simulate human behavior in wildfire simulations for cities to identify the best evacuation routes.

### Conclusion

The development of CMTF-OPT1 software displays a promising use of gradient-based optimization in Coupled Matrix-Tensor Factorization (CMTF) to achieve similar accuracy compared to standard Alternating Least Squares implementations. Mobility data analysis using CMTF can effectively identify hidden components in mobility data using spatial and temporal plots. Furthermore, we have established a streamlined workflow for identifying the optimal rank and performing CMTF decomposition on mobility data.

Future research can test alternative gradient-based optimization techniques in CMTF-OPT1 for faster decompositions. Parallelizing the nonlinear conjugate gradient optimization is also an option. To make CMTF more versatile, future work can adapt TensorLy CMTF-ALS for sparse tensors and matrices to efficiently decompose real-world data.

The main future application of this project is for cities to collect mobility data through different sources and utilize CMTF-OPT1 to uncover hidden components in human mobility. This can be used to simulate human behavior in wildfire simulations for cities to identify the best evacuation routes.

### Dependencies

- Windows 10 Operating System
- Python 3.11.0
- Python random, time, datetime builtin libraries
- NumPy 1.24.1
- SciPy 1.10.0
- Matplotlib 3.6.3
- Pandas 1.5.3
- TensorLy 0.8.0
- Xarray 2023.1.0
- TLViz 0.1.7
- GeoPandas 0.12.2
- Fiona 1.9.1
- Shapely 2.0.1
