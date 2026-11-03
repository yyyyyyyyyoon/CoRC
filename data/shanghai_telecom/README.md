# Shanghai Telecom — not included

This data is required only for the MEC server-selection part of the paper.
Without it, the public repository still supports a partial reproduction of
CoRC on WS-DREAM, including MAE evaluation. The MEC experiment cannot be rerun
end to end from the public files alone because its request/candidate data comes
from an upstream framework and the Shanghai Telecom source data cannot be
redistributed.

The MEC side of the dataset is not redistributed with this repository. Its
terms, on the maintainers' distribution page, state:

> The dataset is available free of charge for educational and non-commercial
> purposes. … **Redistribution of this data to any other third party is not
> permited.**

So it has to be requested from the source:

> https://wangshangguang.github.io/telecom_dataset/

The same page lists the papers it requires you to cite — Li et al. (2022),
Guo et al. (2020) and Wang et al. (2021), which are references [20–22] of our
paper — and asks that results obtained with it be shared back.

## What goes here

Put three files in `files/`:

| file | columns that matter | what it is |
|---|---|---|
| `server_locations.csv` | `latitude`, `longitude`, `server_id`, `matrix_service_id` | the 2,977 base stations, each mapped to one WS-DREAM service |
| `server_status.csv` | `server_id`, `server_time_slot`, `load`, `fail_rate` | load and failure rate per server per five-minute slot |
| `base_samples_candidates_scored.csv` | `sample_id`, `matrix_user_id`, `predicted_lat`, `predicted_lon`, `server_time_slot`, `service_id`, `weight_rt`, `weight_tp`, `weight_ld`, `weight_fr` | one row per request |

If they live somewhere else, point `CORC_MEC_DIR` at the directory holding all
three:

```bash
CORC_MEC_DIR=/path/to/mec python -m pipeline.build_selection_dataset
```

Only `server_locations.csv` carries Shanghai Telecom values unchanged, and it
is the reason none of the three are shipped here. `server_status.csv` is
synthetic, assigned per server per slot by us (§4.2).
`base_samples_candidates_scored.csv` is built upstream, in the edge-computing
framework this work extends: `predicted_lat` and `predicted_lon` are where the
user is predicted to be at request time, and the preference weights come from
the same step.

## Without it

`pipeline.build_selection_dataset` is the only stage that reads any of this,
and it stops with a message saying so. Everything before it — the splits, the
five base predictors, the cross-fitting, the collaborative evidence — runs on
WS-DREAM alone, and the resulting QoS predictions can be evaluated with MAE.
The checked-in summaries under `results/paper/` allow the reported tables and
figure data to be regenerated for inspection; they do not make the private MEC
pipeline fully reproducible.
