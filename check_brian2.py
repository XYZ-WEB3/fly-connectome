"""Эталонная проверка на оригинальном коде Shiu et al. (Brian2):
запах DM1 слева, 5 Гц, 300 мс -> сколько нейронов активно."""
import sys
import pandas as pd
from brian2 import ms, Hz, Network, prefs

sys.path.insert(0, "brain_model")
from model import create_model, poi, default_params as params

prefs.codegen.target = "numpy"
params = dict(params, t_run=300 * ms, r_poi=5 * Hz)
ann = pd.read_csv("data/neuron_annotations.tsv", sep="\t", low_memory=False, usecols=["root_id", "cell_type", "side"])
comp = pd.read_csv("brain_model/Completeness_783.csv", index_col=0)
idx = {r: i for i, r in enumerate(comp.index)}
orn = [idx[r] for r in ann[(ann.cell_type == "ORN_DM1") & (ann.side == "left")].root_id if r in idx]

neu, syn, mon = create_model("brain_model/Completeness_783.csv", "brain_model/Connectivity_783.parquet", params)
pois, neu = poi(neu, orn, [], params)
net = Network(neu, syn, mon, *pois)
net.run(params["t_run"], report="text")
n_active = sum(1 for v in mon.spike_trains().values() if len(v))
print(f"BRIAN2 original: DM1-left 5Hz 300ms -> active neurons: {n_active}")
