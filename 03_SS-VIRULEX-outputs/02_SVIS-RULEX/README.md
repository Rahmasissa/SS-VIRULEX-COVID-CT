# 02_SVIS-RULEX

This folder is the trained SVIS-RULEX artifact cache used by SS-VIRULEX.

Primary trained model:

`covid_ct_run/exact_sequence/outputs/models/03_custom_mobilenetv2_complete_covid_ct.keras`

SS-VIRULEX consumes this folder for the statistical branch: trained MobileNetV2
outputs, statistical feature CSVs, ZFMIS feature sets, and stat-only rule
baselines. The fused SS-VIRULEX branch combines these SVIS-RULEX statistical
features with Med-MICN checkpoint-generated concept probabilities.

The original SVIS-RULEX source repository remains separate at:

`/Users/samehissa/Downloads/XAI-Med-Images-Stat-Visual-Rules-main 2`
