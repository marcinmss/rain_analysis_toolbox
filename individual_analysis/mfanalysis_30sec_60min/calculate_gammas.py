from multifractal_analysis import tm_analysis
from parsivel import pars_read_from_pickle
from stereo3d import stereo_read_from_pickle
from pathlib import Path
from multifractal_analysis.data_prep import prep_data_ensemble
from numpy import concatenate
from pandas import DataFrame
from multifractal_analysis.general import assess_d, assess_gammas, assess_q

parsivel_events_folder = Path(
    "/home/marcio/stage_project/individual_analysis/sprint02/saved_events/parsivel/"
)
stereo_events_folder = Path(
    "/home/marcio/stage_project/individual_analysis/sprint02/saved_events/stereo/"
)


parsivel_events = [
    pars_read_from_pickle(file_path) for file_path in parsivel_events_folder.iterdir()
]

stereo_events = [
    stereo_read_from_pickle(file_path) for file_path in stereo_events_folder.iterdir()
]

# Calculate Gamma S for parsivel
field = concatenate(
    [prep_data_ensemble(event.rain_rate, 2**7) for event in parsivel_events], axis=1
)
tma = tm_analysis(field)
gamass_parsivel = assess_gammas(tma.alpha, tma.c1)


# Calculate Gamma S for parsivel with fluctuations
field = concatenate(
    [
        prep_data_ensemble(event.rain_rate, 2**7, fluc=True)
        for event in parsivel_events
    ],
    axis=1,
)
tma = tm_analysis(field)
gamass_parsivel_wf = assess_gammas(tma.alpha, tma.c1, assess_d(field))

# Calculate Gamma S for stereo
field = concatenate(
    [prep_data_ensemble(event.rain_rate(), 2**7) for event in stereo_events], axis=1
)
tma = tm_analysis(field)
gamass_stereo = assess_gammas(tma.alpha, tma.c1, assess_d(field))


# Calculate Gamma S for stereo with fluctuations
field = concatenate(
    [
        prep_data_ensemble(event.rain_rate(), 2**7, fluc=True)
        for event in stereo_events
    ],
    axis=1,
)
tma = tm_analysis(field)
gamass_stereo_wf = assess_gammas(tma.alpha, tma.c1, assess_d(field))

# Print the answers
df = DataFrame(columns=["Direct Field", "Fluctuations"])
df.loc["parsivel"] = [gamass_parsivel, gamass_parsivel_wf]
df.loc["stereo"] = [gamass_stereo, gamass_stereo_wf]

print(df)
