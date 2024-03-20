from pathlib import Path
from typing import List
from matplotlib.pyplot import figure
from parsivel import ParsivelTimeSeries, parsivel_read_from_pickle
from stereo.dataclass import BASESTEREOSTYLE
from parsivel.dataclass import BASEPARSIVELSTYLE as BASEPARSIVELSTYLE

OUTPUTFOLDER = Path(__file__).parent / "output"
AXESTITLESFONTSIZE = 18
AXESLABELSFONTSIZE = 16
LEGENDFONTSIZE = 20

output_folder = Path(__file__).parent / "output"


def overall_analysis(
    parsivel_events: List[ParsivelTimeSeries],
    stereo_events: List[ParsivelTimeSeries],
):
    fig = figure(dpi=300, figsize=(5, 4), layout="constrained")

    # dsd per class
    ax = fig.add_subplot(1, 1, 1)
    x, _ = parsivel_events[0].get_nd3()
    y = sum([event.get_nd3()[1] for event in parsivel_events]) / len(parsivel_events)
    ax.plot(x, y, **BASEPARSIVELSTYLE, label="Parsivel")
    y = sum([event.get_nd3()[1] for event in stereo_events]) / len(stereo_events)
    ax.plot(x, y, **BASESTEREOSTYLE, label="3D Stereo")
    ax.set_ylabel("$N(d).d^{3}$", fontsize=AXESLABELSFONTSIZE)
    ax.set_xlabel("Diameter $(mm)$", fontsize=AXESLABELSFONTSIZE)
    ax.set_xbound(0, 6)

    ax.legend(fontsize=12, frameon=False)
    fig.savefig(OUTPUTFOLDER / "mean_dsd_of_all_events.png")


if __name__ == "__main__":
    pars_folder = Path(
        "/home/marcio/stage_project/data/saved_events/Set01/events/parsivel/"
    )
    stereo_folder = Path(
        "/home/marcio/stage_project/data/saved_events/Set01/events/stereo_converted/"
    )
    parsivel_events = [
        parsivel_read_from_pickle(file_path)
        for file_path in sorted(pars_folder.iterdir())
    ]
    stereo_events = [
        parsivel_read_from_pickle(file_path)
        for file_path in sorted(stereo_folder.iterdir())
    ]
    print("READ THE DATA FOR BOTH DEVICES")
    overall_analysis(parsivel_events, stereo_events)
    print("DONE.")
