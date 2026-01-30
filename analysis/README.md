# Instructions for parsing physical layer metrics

### Required files in Phys-layer-unparsed folder
- '5g experiments - Runs.csv' - Go to [5g experiments on Google Sheets](https://docs.google.com/spreadsheets/d/1ma8kws76cSF5ChryjkMLhKMKvKFhVOoOdpalL2RJMac/edit?usp=sharing) and download the latest version of the Runs sheet as a csv file (or make sure that all Run IDs in folder have entries in the csv).
- oaibox.telemetry_\[Date\].json - First add new set of telemetry to '5G Experiment Data/oaibox-telemetry/zipped/' and unzip data that should be parsed. Only add oaibox.telemetry_\[Date\].json into Phys-layer-unparsed folder, NOT oaibox.ue.telemetry_\[Date\].json, which has different schema that is not set up to be parsed. (It's the same data but collected twice per second rather than once and contains a lot of redundancy - it's collected per UE, not per time of measurement, so it has redundant heading info.)
- All Flower FL data (folders named by Run ID) from server computer for runs that may have phys layer data in the oaibox telemetry json. The folders can be renamed to the correct format using rename_folders.py from the command line.

### Parse
1. Add all relevant files into '5G Experiment Data/Phys-layer-unparsed/'. (Change the filepath in oaibox_telemetry.py and use '5G Experiment Data/Phys-layer-unparsed demo/' to preview functionality when new data is not available. These folders are copies, so you can delete the phys_layer subfolders and re-parse.)
2. Run oaibox_telemetry.py with call to parse() uncommented.
3. Check that trial data folders have phys_layer subfolders. If there is none, data may need to be saved from OAIBOX first and then parsed. Move folders with phys layer metrics to FedAvg. Move oaibox.telemetry_\[Date\].json to oaibox-telemetry/unzipped/. Delete '5g experiments - Runs.csv'.