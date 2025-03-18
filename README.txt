
This directory contains Python code for estimating drop size distributions. The code is not executable in its current form because it requires additional files that cannot be anonymized. These are:
- DSD/dataset/attribute_data.py
- DSD/dataset/attribute_names.py
- DSD/dataset/data_id.py
- DSD/dataset/rawdata2.py
- {path_to_datetime_dir}/202106_datetime_10MB.txt
- {path_to_datetime_dir}/202107_datetime_10MB.txt
- {path_to_datetime_dir}/202206_datetime_10MB.txt
- {path_to_datetime_dir}/202106_datetime_Kumagaya.txt
- url_opener.py

In addition to DSD estiamtion code, thid direcoty contains a python code for creating lookup tables for radar observables assuming lognormal distribution:
computation_lognormal_with_elevation.py

disdrodb_work directory stores the analysis codes (ipynbs) for disdrodb (https://disdrodb.readthedocs.io/en/latest/).
analysis directory stores the analysis routines (ipynbs) used for DSD estimate.
