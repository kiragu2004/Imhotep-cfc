# Imhotep CfC Hazard API

Decision-support hazard forecasts for Kenyan counties, using a 19-neuron CfC network.

Targets: dry_hot, heavy_rain, fire_weather
Input: last 30 days of NASA POWER weather (5 features)
Output: calibrated probability for each hazard

## Endpoints
- GET  /               landing page
- GET  /health         health check
- GET  /counties       list trained counties
- POST /predict/{county}  predict hazards

## Disclaimer
Decision support only. Not an official warning.
Consult KMD, ICPAC, or county offices for life-safety decisions.