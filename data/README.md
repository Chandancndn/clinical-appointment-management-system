# Data: sources, exact files, and how to rebuild everything

Three public datasets, each with one job. **No raw file is ever committed**: they live in `data/raw/`, which is
gitignored (CLAUDE.md hard rule 5). The app never reads them either; the app is seeded with synthetic data only
(`db/seed_synthetic.py`, hard rule 4).

```
data/raw/
  kaggle/KaggleV2-May-2016.csv
  openml/medical_appointment_43617.arff
  hangu/Data.csv            (the other Hangu files can sit beside it; only Data.csv is read)
```

| Job | Dataset | Where to get it | Exact file used |
|---|---|---|---|
| Primary: develop and report no-show models | Kaggle, "Medical Appointment No Shows" (`joniarroba/noshowappointments`) | https://www.kaggle.com/datasets/joniarroba/noshowappointments (needs a Kaggle login) | `data/raw/kaggle/KaggleV2-May-2016.csv` |
| Second: check that findings hold | OpenML, "Medical-Appointment", data id 43617 | https://www.openml.org/d/43617 | `data/raw/openml/medical_appointment_43617.arff` |
| Consultation times, for the simulation only | Hangu open data (Feng et al.) | https://github.com/fenghaolin/HanguData, also Zenodo, doi [10.5281/zenodo.7484205](https://doi.org/10.5281/zenodo.7484205) | `data/raw/hangu/Data.csv` |

## Download steps

1. **Kaggle.** Sign in, open the dataset page, download and unzip. The zip (`archive.zip`) holds one file,
   `KaggleV2-May-2016.csv`; put it in `data/raw/kaggle/`. With the Kaggle API:
   `kaggle datasets download -d joniarroba/noshowappointments`.
2. **OpenML.** On https://www.openml.org/d/43617 use the download button for the ARFF file. The browser saves it
   as plain `dataset` with no extension; **rename it** to `medical_appointment_43617.arff` and put it in
   `data/raw/openml/`. (`sklearn.datasets.fetch_openml(data_id=43617)` fetches the same data, but the pipeline reads
   the ARFF file.) Listed licence: GPL 2.
3. **Hangu.** Download the repository as a zip from GitHub (or `git clone`) and copy `Data.csv` into
   `data/raw/hangu/`, together with `LICENSE` and `CITATION.cff` for the record. Licence: CC BY-NC-SA 4.0
   (coursework use with credit).

My downloads were made on 2026-10-06 (from the file timestamps). Please record your own date if you re-download.

## Check that your copy is the same

```bash
python -m ml.src.data
```

prints, for each raw file, its row count, column count, size and SHA-256. Compare the row counts with
`rows_raw` in `results/dataset_summary.csv`, and the hashes with the `data_files` entries in
`results/manifest.json`. They must match; if not, you have a different version of the data.

## Which Hangu file, and why

The Hangu repository holds four CSVs. Two of them contain `ServTime` (consultation length **in seconds**):

* `Data.csv`: **the file used.** One row per consultation, with no missing or non-positive `ServTime`. Its row count
  is the 6,637 consultations from 381 half-day sessions that CLAUDE.md describes (`rows_raw` and `sessions` in
  `results/dataset_summary.csv`). Its smallest consultation is `service_seconds_min` in that file: nothing shorter
  than that appears.
* `Merged_ServiceTime.csv`: the same consultations **plus extra rows** that `Data.csv` does not contain, including
  missing values, a zero, and implausibly short and long times. It looks like the unfiltered version that the
  dataset's authors cleaned to produce `Data.csv`. Not used.
* `Raw_1.csv` (no `ServTime`) and `Raw_2.csv` (patient addresses) are not used.

**Limitation to state in the report:** because `Data.csv` appears to have been filtered upstream, the shortest
consultations are missing, so the simulated service-time distribution has an artificial floor at
`service_seconds_min`. The size of that floor comes from `results/dataset_summary.csv`, not from this file.

## Coding traps (positive class is always `no_show = 1`)

| Dataset | Source column | Meaning | In our code |
|---|---|---|---|
| Kaggle | `No-show` | `Yes` = the patient **missed** the appointment | `no_show = 1` for `Yes` |
| OpenML | `show` | `1` = the patient **attended** | `no_show = 1 - show` |

Other things that differ between the files, all handled in `ml/src/data.py` and logged in
`results/cleaning_log.csv`:

* **Kaggle** has two column typos (`Hipertension`, `Handcap`), a float `PatientId` (a few values are not whole numbers
  and are dropped as malformed), and `AppointmentDay` stored at 00:00 with `ScheduledDay` carrying a time, so the
  date rule compares calendar dates, not timestamps.
* **OpenML** column names are Spanish (`edad` age, `sexo` sex with 1 = male and 2 = female, `latencia` lead days,
  `canal` booking channel, `tipo` appointment type, ...). It has **no calendar dates, only month, weekday and hour
  parts**, so its holdout is "the last month". It has **no patient ID**, so duplicates cannot be told from similar
  patients. The six cosine columns (`*_c`) repeat the discrete ones and are dropped. Origin and country are not stated.
* **Hangu** times are in seconds and are converted to minutes.

## Cleaning, splits and the E1 profile

```bash
python -m ml.src.e1_profile
```

reads the three files and writes `results/cleaning_log.csv`, `results/dataset_summary.csv`,
`results/e1_eda_rates.csv`, `results/figures/e1_*.png` and the matching entries in `results/manifest.json`.
Every number the report quotes about these datasets comes from those files.

## Citations

* Hangu: Feng, H., et al. (2023). Data 8(3), 47. https://doi.org/10.3390/data8030047 (the repository's own README
  still cites the 2022 working-paper version).
* Kaggle: "Medical Appointment No Shows", `joniarroba/noshowappointments`. Check the licence on the Kaggle page.
* OpenML: "Medical-Appointment", data id 43617.
