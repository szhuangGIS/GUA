# Urbanization is linked to uneven amplification of lake algal blooms worldwide

This repository provides the analysis and plotting code, together with processed data, for a study of lake algal bloom changes associated with watershed urbanization. The study examines the direction, magnitude and geographical variation of these changes across 2,216 matched groups of lakes in urbanizing catchments and their reference lakes.

The release covers calculation of urbanization effect and significance assessment, environmental model training and evaluation, and the preparation of main Figures 1–4. It contains analysis tables and figure source data. The accompanying manuscript and Supplementary Material describe the underlying observations, data sources and methods.

## Bloom detection

Lake algal blooms were detected using the Landsat-based algorithm and original Google Earth Engine (GEE) code released by [Hou et al. (2022)](https://doi.org/10.1038/s41561-021-00887-x). Their implementation is available through the [GEE bloom-detection code](https://code.earthengine.google.com/325dfdde02b381419bb81a71bee769d7).

For further details on the generation of satellite-derived bloom observations, including the detection algorithm, image screening, masking and validation, please refer to their paper and its Supplementary Information.

**Reference:** Hou, X. et al. Global mapping reveals increase in lacustrine algal blooms over the past decade. *Nature Geoscience* **15**, 130–134 (2022). [https://doi.org/10.1038/s41561-021-00887-x](https://doi.org/10.1038/s41561-021-00887-x).

## Repository structure

```text
.
├── README.md
├── Code/               Python analysis and plotting scripts
└── Data/               Excel input tables
    └── Map_Boundary/   Map boundary shapefile and supporting files
```

## Software environment

The code is written in **Python**. The reference environment is **Windows with Python 3.10.18**. The package versions below match the local environment used to check the release.

The main dependencies are NumPy and pandas for data processing; SciPy, scikit-learn, LightGBM and statsmodels for statistical and model analyses; Matplotlib for plotting; openpyxl for Excel input; and PyShp, pyproj and Shapely for maps. IPython supports table display in the analysis scripts.

One installation option, using Conda, is:

```bash
conda create -n lake-urbanization python=3.10.18 -y
conda activate lake-urbanization
python -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.15.3 scikit-learn==1.7.1 lightgbm==4.6.0 statsmodels==0.14.5 matplotlib==3.10.0 openpyxl==3.1.5 pyshp==2.3.1 pyproj==3.6.1 shapely==2.0.5 ipython==8.30.0
```

## Code

| Script | Overview |
| --- | --- |
| [01_Calculate_UE_and_BH_Test.py](Code/01_Calculate_UE_and_BH_Test.py) | Examines covariate distributions and balance before and after reference selection, calculates UE, and applies Benjamini–Hochberg (BH) correction to the Mann–Kendall (MK) p-values. |
| [02_Train_UE_Models.py](Code/02_Train_UE_Models.py) | Organizes samples and trains LightGBM models for frequency UE and MBEP UE using nine environmental and urban-activity predictors. Includes preprocessing, parameter tuning, nested cross-validation, model accuracy assessment, SHAP and permutation-importance analyses. |
| [03_Plot_Figure1.py](Code/03_Plot_Figure1.py) | Plots the spatial distribution and significance of UE, together with summaries across latitude, lake size, continents and climate zones. |
| [04_Plot_Figure2.py](Code/04_Plot_Figure2.py) | Plots urban-minus-reference differences in pre/post log changes and the evolution of bloom differences relative to the identified expansion year. |
| [05_Plot_Figure3.py](Code/05_Plot_Figure3.py) | Plots country-level UE patterns and comparisons across long-term GDP growth groups, including lake-level distributions and country-level summaries. |
| [06_Plot_Figure4.py](Code/06_Plot_Figure4.py) | Plots relative feature importance and SHAP dependence curves for the two bloom metrics. |

## Data

All tabular inputs are supplied as `.xlsx` files. Figure source workbooks organize the data into worksheets named by panel, such as `Fig1a`; the other workbooks use a worksheet named `Data`.

| File or directory | Overview |
| --- | --- |
| [01_Covariate_Distributions.xlsx](Data/01_Covariate_Distributions.xlsx) | Values and weights for the distributions of lake area, mean depth, elevation and agricultural nitrogen and phosphorus backgrounds. Includes urban lakes and reference lakes before and after selection. |
| [01_UE_Matched_Groups.xlsx](Data/01_UE_Matched_Groups.xlsx) | The 2,216 matched groups, including urban-lake identifiers and coordinates, reference-lake identifiers, urban and reference slopes, UE, and raw and BH-adjusted MK p-values for both bloom metrics. Multiple reference identifiers are separated by commas. |
| [02_SHAP_Analysis.xlsx](Data/02_SHAP_Analysis.xlsx) | Model input data containing urban-lake identifiers, the two bloom UE outcomes, and UE values for nine predictors: temperature, nighttime lights, vapour pressure deficit, precipitation, runoff, wind speed, radiation, soil moisture and leaf area index. |
| [03_Figure1_Source_Data.xlsx](Data/03_Figure1_Source_Data.xlsx) | Panel-specific data for UE maps, significance summaries and comparisons by geographical and lake characteristics. |
| [04_Figure2_Source_Data.xlsx](Data/04_Figure2_Source_Data.xlsx) | Urban and reference log changes for the spatial comparisons, plus processed mean differences and standard errors by lake-size class and year relative to the identified expansion year. |
| [05_Figure3_Source_Data.xlsx](Data/05_Figure3_Source_Data.xlsx) | Country-level and lake-level UE summaries used in the GDP growth comparisons, maps and country rankings. |
| [05_Figure3_Country_Background.xlsx](Data/05_Figure3_Country_Background.xlsx) | Country ISO codes and GDP growth categories used to draw the background of the Figure 3 maps, including countries outside the lake analysis sample. |
| [06_Figure4_Source_Data.xlsx](Data/06_Figure4_Source_Data.xlsx) | Mean absolute SHAP values and predictor–SHAP pairs used to plot feature importance and dependence curves. |
| [Map_Boundary/](Data/Map_Boundary/) | The `WB_GAD_ADM0_Income` map boundary shapefile and its supporting files, used by the Figure 1–3 scripts. |

## Running the scripts

Download or clone the repository and retain the `Code/` and `Data/` directory structure. From the repository root, run the desired script:

```bash
python Code/01_Calculate_UE_and_BH_Test.py
python Code/02_Train_UE_Models.py
python Code/03_Plot_Figure1.py
python Code/04_Plot_Figure2.py
python Code/05_Plot_Figure3.py
python Code/06_Plot_Figure4.py
```

Input paths are resolved relative to each script's location. The scripts can be run independently.

Results are printed or displayed during execution. Figures are shown with Matplotlib, and the scripts do not automatically write output files. If execution pauses at a figure window, close the window to continue. 

Model training is more computationally demanding than plotting. Parallel execution in `02_Train_UE_Models.py` can be reduced through `OUTER_WORKERS` and `MODEL_THREADS` if needed.
