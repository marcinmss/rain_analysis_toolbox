from pathlib import Path
from matplotlib.pyplot import figure
from matplotlib import colormaps

from stereo import stereo_read_from_pickle
from individual_analysis.analysis_variables import (
    AXESLABELSFONTSIZE,
    FIGURESPECS,
    LEGENDSPECTS,
)

STEREOFULLEVENTFILE = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/stereo_full_event.obj"
)
OUTPUTFOLDER = Path(__file__).parent / "output/"
CMAP = colormaps["tab20b"]

if __name__ == "__main__":
    print("Reading the full event stereo.")
    stereo_event = stereo_read_from_pickle(STEREOFULLEVENTFILE)
    print("Splitting the full event in sections")
    sections = stereo_event.split_by_distance_to_sensor()
    del stereo_event

    print("Calculating DSD and plotting the results")
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()
    ax.set_ylabel("$N(d).d^3$", fontsize=AXESLABELSFONTSIZE)
    ax.set_xlabel("Diameter $(mm)$", fontsize=AXESLABELSFONTSIZE)
    for nsection, section in enumerate(sections, 1):
        converted_section = section.convert_to_parsivel()
        style = {"label": f"section_{nsection}", "color": CMAP((nsection + 1) / 10)}
        x, y = converted_section.get_nd3()
        ax.plot(x, y, color="black", linewidth=2.1)
        ax.plot(x, y, **style)

    ax.set_xbound(0, 3)
    ax.legend(**LEGENDSPECTS)
    fig.savefig(OUTPUTFOLDER / "sda_dsd_of_sections.png")
    print("Done.")
