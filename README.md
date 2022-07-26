
# ml_clustering_core

This is a package for clustering of core data using machine learning clustering techniques (kmean, dbscan, som, ect.).

Currently, the following methods are included:

Kmean clustering 

Spectral clustering 

Gaussian mixture (GM) clustering 

C-fuzzy clustering DBSCAN clustering 

Self Organized Map (SOM) clustering

There are few steps of data QC and pre-processing applied before implemneting clutering. Those are:

Remove nulls 

Remove outliers 

Transforming data to [0 1] 

Exploratory Data Analysis (EDA) : Scatter plots, Density plots, Correlation Study, PCA, etc.

Contribution are welcome to include new data sets and other types of clustering methods. Converting the codes to other programming languages are also welcome.

Please make sure to cite this paper if use these resources here.

http://www.irpga-journal.ir/article_143421.html?lang=en

Salahshoor, A., Gaeini, A., Shahin, A., Karami, M., Reservoir Characterization Clustering Analysis to Identify Rock Type Using KMEANS Method in South-West Iranian Oil Field, 2022, Journal of Petroleum Geomechanics, Vol. 4, Issue 2, Pg. 42-55, http://www.irpga-journal.ir/article_143421.html?lang=en



Determination of rock types is of special importance in the construction of static and dynamic models of hydrocarbon reservoirs. Estimating the properties of reservoir rocks increases the accuracy in predicting the amount of reservoir storage and its performance. Numerous models have been proposed by experts to determine the types of reservoir rocks. But most of the proposed models are based on conventional methods based on engineering and geology of carbonate reservoir rocks. Therefore, using a machine learning method to determine rock species in comparison with previous methods and comparing its efficiency and performance with other methods seems necessary. In this study, core and log data in maroon oil reservoir after preparation were match using Dynamic Time Series (DTW) technique for depth matching. The brain data were then clustered by the non-supervised machine learning method. The kernel data clustering process was also performed by conventional model-based methods such as flow zone index method (FZI) and Winland. Then, the clustering results were validated and compared with kmeans, FZI and Winland methods by having the lithology information of the logs. The kmeans method with a 93.5% accuracy criterion succeeded in performing the highest cluster resolution, which showed that the kmeans data-based machine learning method is a suitable alternative to conventional model-based methods for clustering rock typing.
