# Covid Mapping

This code maps the current case rates among other statistics for the UK.  Data is collected from the UK Government API (https://coronavirus.data.gov.uk/) and uses plotly to show the data on an interactive, animated graph.

## Generating the plots

The LTLA data was gathered from 2 sources which were fused together as neither contained usable completed data.  Download the govt ltla boundaries from [2018](https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2018-boundaries-gb-bfc/explore) and [2021](https://geoportal.statistics.gov.uk/datasets/local-authority-districts-may-2021-uk-bfe/explore) then run `convert_shp.py` to generate the ltla dataframes that are used by `covidDataPlot.py`.

Plotly will open a new browser tab in your default browser with the interactive plots.