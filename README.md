# AWSSDR T30 estimation

Room reverberation time describes how long sound takes to fade. AWSSDR estimates T30 from recorded speech without measuring a room impulse response at inference time. It converts speech into spectral decay rates, then learns which time frames and frequency bands carry useful information about the room. This package provides the pipeline for RIR labelling, training-data and feature generation, model training, and checkpoint averaging. It does not include pretrained weights or a standalone inference command.

## Results reported in the study

The AWSSDR report (10 November 2026) evaluates four models trained on ISO 3382 T30 labels: Base and DRR-boosted training pools, each with a 4 kHz or 6 kHz front end. On the 4,500 ACE EVAL utterances, scored against the official FB T30 ISO reference, the results are:

| Model | MSE (s²) | Bias (s) | MAE (s) | Pearson ρ |
|---|---:|---:|---:|---:|
| Base 4k | 0.0226 | −0.021 | 0.1072 | 0.935 |
| Base 6k | **0.0196** | −0.029 | **0.1022** | **0.949** |
| DRR-boosted 4k | 0.0312 | −0.020 | 0.1138 | 0.906 |
| DRR-boosted 6k | 0.0271 | −0.024 | 0.1129 | 0.926 |

![ACE EVAL confusion matrices for the four AWSSDR models](docs/images/ace-eval-confusion-matrices.png)

*ACE EVAL confusion matrices from Figure 5 of the AWSSDR report: official FB T30 ISO value on the x-axis and estimated T30 on the y-axis.*

On a free-field reference microphone in real rooms, the models closely tracked measured T30 (MSE 0.0185–0.0256 s² across the four models). Base models underestimated T30 on an array microphone and an iPhone, especially in longer rooms. Training with raised direct-to-reverberant-ratio (DRR) RIR copies substantially reduced this bias. None of the models produced usable estimates from VoIP call audio. These are results of the report's trained models and recordings, not metrics produced by installing this package; the study covers one training run per model and 18 room conditions at three sites.

## Pipeline

Each command reads the folder produced by the preceding step:

| Step | Command | Module |
|---|---|---|
| 0. T30 labels of an RIR pool | `awssdr-t30-labels` | `awssdr.acoustics.t30_labels` |
| Optional: DRR-boosted RIR copies | `awssdr-drr` | `awssdr.acoustics.drr_augmentation` |
| Optional: combine RIR folders into one pool | `awssdr-pool` | `awssdr.acoustics.pool` |
| 1. Noisy reverberant utterances | `awssdr-utterances` | `awssdr.data.utterances` |
| 2. Decay-rate features | `awssdr-decay-rates` | `awssdr.data.decay_rates` |
| 3. Training | `awssdr-train` | `awssdr.training.train` |
| 4. Averaging epoch checkpoints | `awssdr-average` | `awssdr.training.average_checkpoints` |

## Repository layout

```text
src/awssdr/
  dataset_layout.py         file and folder names the steps share, sample rate
  acoustics/                t30_labels.py, drr_augmentation.py, pool.py, acoustic_features.py
  data/                     utterances.py, decay_rates.py
  models/                   rt_est.py (the RT_est network)
  training/                 train.py, average_checkpoints.py
tests/                      mirrors src/awssdr
```

## Tests

```bash
python -m unittest discover -s tests -t .
```

## Data setup

No data paths are built into the code: every command takes its folders as options. Put the speech and noise corpora under one folder, `<ROOT>` (passed as `awssdr-utterances --root`), in this layout. Symbolic links to corpora stored elsewhere work as well.

```text
<ROOT>/
  Speech/EARS/<speaker>/*.wav                       e.g. p001/rainbow_01_regular.wav
  Speech/SET-2/GSP/<word>/*.wav                     + validation_list.txt, testing_list.txt
  Speech/SET-2/Qualcomm/<keyword>/<speaker>/*.wav
  Speech/SET-2/ST_AED/*.wav                         f0001_us_f0001_00001.wav, ...
  Speech/timit/timit/dr<N>-<speaker>/*.wav          e.g. dr1-fvmh0/sa1.wav
  Noise/DEMAND/<ENV>_16k/<ENV>/ch01.wav
  Noise/MS-SNSD/AirConditioner_1.wav ... AirConditioner_10.wav
  Eval_data/AID/wavs/<class>_<n>_SH_MKH800.wav
```

| Folder | Corpus and source | Notes |
|---|---|---|
| `Speech/EARS` | EARS, [github.com/facebookresearch/ears_dataset](https://github.com/facebookresearch/ears_dataset) | Every speaker folder present is used. Our datasets use only `p001`-`p010` of the 107 speakers: link only those to reproduce them. CC BY-NC 4.0 |
| `Speech/SET-2/GSP` | Google Speech Commands v0.02, [speech_commands_v0.02.tar.gz](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) | Unpack the whole archive; its `validation_list.txt` and `testing_list.txt` exclude those clips. CC BY 4.0 |
| `Speech/SET-2/Qualcomm` | Qualcomm Keyword Speech Dataset, [qualcomm.com/developer/software/keyword-speech-dataset](https://www.qualcomm.com/developer/software/keyword-speech-dataset) | Unpacks as `<keyword>/<speaker>/*.wav`. Research use only |
| `Speech/SET-2/ST_AED` | Free ST American English Corpus (ST-AEDS-20180100_1), [openslr.org/45](https://www.openslr.org/45/) | Put the WAV files directly in the folder. CC BY-NC-ND 4.0 |
| `Speech/timit/timit` | TIMIT corpus sample (16 speakers, 160 recordings) from the NLTK data package `timit` (`python -m nltk.downloader timit`) | The free sample, not the full LDC corpus; place its `dr<N>-<speaker>` folders here. CC BY-NC-SA |
| `Noise/DEMAND` | DEMAND, [zenodo.org/records/1227121](https://zenodo.org/records/1227121) | Unpack each 16 kHz archive `<ENV>_16k.zip` into `<ENV>_16k/`. The 12 environments in `DEMAND_ENVS` (`awssdr.data.utterances`) are required. CC BY-SA 3.0 |
| `Noise/MS-SNSD` | MS-SNSD, [github.com/microsoft/MS-SNSD/tree/master/noise_train](https://github.com/microsoft/MS-SNSD/tree/master/noise_train) | Copy only `AirConditioner_1.wav` to `AirConditioner_10.wav`, flat. Licences per the upstream README |
| `Eval_data/AID` | AID: Open-source Anechoic Interferer Dataset, [zenodo.org/records/6974033](https://zenodo.org/records/6974033) | Only the `SH_MKH800` recordings of the 12 classes in `TRANSIENT_CLASSES` are used. CC BY-NC-SA 4.0 |

Several of these licences are non-commercial or research-only; check them before any use beyond research.

**RIRs** are a separate folder per pool (`--rir-dir`): mono 16 kHz `*.wav` files plus `T30_ground_truth.csv` (columns `filename`, `t30` in seconds). `awssdr-t30-labels` makes the CSV; `awssdr-drr` and `awssdr-pool` add DRR copies.

**Dev set:** `awssdr-train --dev` needs decay-rate features of the ACE corpus (Eaton et al.) dev recordings. No command builds them yet; a feature folder from the previous pipeline (labels in `T60/`) works.

Check a setup without writing anything:

```bash
awssdr-utterances --root <ROOT> --rir-dir <RIR_DIR> --dry-run
```

A missing corpus, noise recording or transient class stops with an error naming the file or folder.

## Step 0: T30 labels

```bash
awssdr-t30-labels --rir-dir <RIR_DIR>
```

Measures the broadband Lundeby T30 of every mono `*.wav` directly in `<RIR_DIR>` and writes `<RIR_DIR>/T30_ground_truth.csv` (columns `filename`, `t30` in seconds), the file step 1 reads. It never replaces an existing CSV. If any RIR fails (unreadable, not mono, no valid T30), it lists all failures and writes nothing.

Pools with DRR-augmented copies (such as `mixed_v13`) give each copy its original's label; that is decided when the pool is built, so their CSVs come from the pool's build script, not from this one.

## Optional: DRR-boosted copies

```bash
awssdr-drr --rir-dir <POOL> --out <NEW_DIR> [--drr-lo 6 --drr-hi 16]
```

Writes one copy per suitable RIR of `<POOL>` with its direct sound boosted to a target DRR in `[max(drr-lo, source DRR + 1), drr-hi]` dB (`<stem>__drr<XXpX>.wav`), their labels in `T30_ground_truth.csv` (each copy keeps its source's T30, since the tail is untouched) and `metadata.csv` (one row per source, including why a source got no copy). Only the copies are written; combine them with the originals when building a training pool. `--out` must not exist yet. Targets are seeded by row number in the source CSV, so the same pool always gives the same copies.

## Optional: combine RIR folders into one pool

```bash
awssdr-pool --rir-dirs <POOL> <DRR_COPIES> --out <NEW_POOL>
```

Links every RIR listed in each folder's `T30_ground_truth.csv` into `<NEW_POOL>` and writes one `T30_ground_truth.csv` with all of them, the layout step 1 reads. Typical use: an RIR pool plus its DRR copies from `awssdr-drr`. File names must be unique across the folders; every check runs before `<NEW_POOL>` (which must not exist) is created. The links point at the original files, so keep those in place.

## Step 1: make utterances

```bash
awssdr-utterances \
  --root <ROOT> --rir-dir <RIR_DIR> --out <UTTERANCES> --jobs 4
```

`--out` must not exist yet (default: `<ROOT>/training_utterances`).

Check first without writing anything: add `--dry-run`. Small test run: add `--sources gsp --limit-rirs 1` (9 utterances).

| Option | Meaning |
|---|---|
| `--sources` | speech corpora to use: `ears gsp qualcomm st_aed timit` (default all) |
| `--ears-passes N` | keep the first N of the 3 EARS passes (default 3); other passes keep their `pNN` numbers, so a smaller N gives a subset of the full run |
| `--noise-types` | `Ambient Fan Babble` (default all) |
| `--limit-rirs N` | use only the first N RIRs (default all) |
| `--seed` | random seed (default 42) |
| `--jobs` | worker processes (default 1) |
| `--dry-run` | plan and validate, write nothing |

**What it makes.** Speech is reverberated with an RIR, noise is reverberated with the same RIR, and both are mixed at 0, 10 and 20 dB SNR. Seven speech passes are used (EARS three times, the other four corpora once each); `--ears-passes` keeps fewer EARS passes.

Ambient and Fan noise are shaped from recordings, with transient events (about one per 20 s). Babble is 4-7 EARS talkers other than the target speaker, plus a quiet Ambient bed.

**Output:**

```text
<UTTERANCES>/
  clips/<sample_id>.wav     one file per utterance, named <pass>_<corpus>__<rir>__<noise>__<snr>dB
  manifest.csv              one row per utterance: T30 label, input files, SNR, levels, noise details
  run.json                  options and library versions
  COMPLETE.json             written last
```

Full default run: about 115 GB of WAVs.

## Step 2: compute decay rates

```bash
awssdr-decay-rates \
  --utterances <UTTERANCES> --out <FEATURES> --jobs 4
```

`--utterances` is the step 1 folder (any name). It must contain `COMPLETE.json`. `--out` must not exist yet. Use `--front-ends 6k` or `4k` to compute only one (default both).

**Output:**

```text
<FEATURES>/
  v1-6k/
    decayRates/<sample_id>.npy    float32, shape (40, n_frames)
    T30/<sample_id>.npy           the label: T30 in seconds
    dataset.pkl                   list of (sample_id, n_frames)
  v1-4k/                        same layout, 4 kHz front end
  manifest.csv
  run.json
```

Full default run: about 36 GB for both front ends.

## Step 3: train

```bash
awssdr-train --train <FEATURES>/v1-6k --dev <DEV_FEATURES> --out <RUN_DIR>
```

`--dev` is a feature folder in the same layout (for example the ACE dev pool), scored after every epoch. Feature folders made before the label folder was renamed (`T60/` instead of `T30/`, e.g. the current ACE dev sets) are still read. `--out` must not exist yet. The defaults are the recipe of the delivered DIN models: 100 epochs, batches of 16 sorted by length, gradients summed over 16 batches per step and clipped to 1, Adam (lr 1e-3, weight decay 1e-3), learning rate x0.95 per epoch after epoch 50, seed 42. Each is an option (`--help`).

**Output:**

```text
<RUN_DIR>/
  latest_weights.pyt                    weights after the last epoch
  epoch_checkpoints/epoch<N>_weights.pyt  every epoch from --save-epochs-from (default 40), for averaging
  train_log.csv                         per epoch: train loss and dev bias, MAE, MSE, Pearson rho
  run.json                              options
```

The delivered model is the average of epochs 90-100 (step 4). Weight files load with `RT_est(num_channels=40, fc_dim=512).load(path)`.

## Step 4: average epochs

```bash
awssdr-average --run <RUN_DIR> --first 90 --last 100
```

Writes `<RUN_DIR>/swa_<first>_<last>_weights.pyt`, the elementwise mean of `epoch_checkpoints/epoch<first..last>_weights.pyt` (default 90-100, the delivered models). Every epoch in the range must have been saved by `train.py --save-epochs-from`. The network has no batch normalisation, so nothing needs recomputing after averaging.

## Credits

- `src/awssdr/models/rt_est.py` is `models/Model_structure.py` of the authors' AWSSDR code (S. Kim and H. Kim), copied unchanged and included under MIT with their permission.
- `src/awssdr/acoustics/acoustic_features.py` (Lundeby T30 fit with pyfar/pyrato) is copied unchanged from the project's `scripts/acoustic_features.py`.

## How to install and use

Use Python 3.13 for the tested dependency versions (`pyproject.toml` requires Python 3.11 or newer). From the repository root, install the pinned dependencies and the package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -t .
```

Prepare the [speech and noise corpora](#data-setup) and an RIR folder, then run the pipeline. Each output folder must be new; replace the placeholders with your paths:

```bash
awssdr-t30-labels --rir-dir <RIR_DIR>
awssdr-utterances --root <ROOT> --rir-dir <RIR_DIR> --out <UTTERANCES> --dry-run
awssdr-utterances --root <ROOT> --rir-dir <RIR_DIR> --out <UTTERANCES> --jobs 4
awssdr-decay-rates --utterances <UTTERANCES> --out <FEATURES> --jobs 4
awssdr-train --train <FEATURES>/v1-6k --dev <DEV_FEATURES> --out <RUN_DIR>
awssdr-average --run <RUN_DIR> --first 90 --last 100
```

`<DEV_FEATURES>` must already contain development-set features in the same layout as the training features. The [pipeline steps](#pipeline) describe the optional DRR augmentation and RIR pooling commands and each step's inputs and outputs.
