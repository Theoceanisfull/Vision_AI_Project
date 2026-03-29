# Dataset Setup

This project uses the ASL-DVS dataset.

Recommended sharing workflow:

- Push the code in this repo to GitHub.
- Upload `ICCV2019_DVS_dataset.zip` to SharePoint or your shared drive.
- Have teammates download the zip locally, then run the setup script from this repo.

## What To Share

- Code: this repository
- Dataset archive: `ICCV2019_DVS_dataset.zip`

The archive is about `5.5 GB`.
After extraction, the dataset is about `19 GB`.
The extracted dataset contains `100,800` `.mat` files across `24` class folders.

## Important Detail

The shared archive is a zip that contains `24` inner per-class zip files.
Because of that, teammates should use the setup script instead of only clicking "Extract All" once.

## Teammate Setup

1. Clone this repository.
2. Download `ICCV2019_DVS_dataset.zip` from SharePoint to any local folder, or use the script to download it directly from OpenI.
3. Run the setup script from the repo root.

If downloading directly with the script, install the OpenI client first:

```bash
python -m pip install --user openi
python scripts/setup_asldvs_for_tonic.py
```

This will download `ICCV2019_DVS_dataset.zip` and extract it under `data/ASLDVS`.

Example:

```bash
python3 Vision_AI_Project/scripts/setup_asldvs_for_tonic.py \
  --save-to Vision_AI_Project/data \
  --download-dir /path/to/folder/that/contains/the/zip \
  --skip-download \
  --skip-verify
```

This will extract the archive and create:

```text
Vision_AI_Project/data/ASLDVS
```

## Optional Verification

If `tonic` is installed, teammates can let the script verify the dataset at the end by removing `--skip-verify`.

## Notes

- The script looks recursively under `--download-dir`, so the zip does not need to be in the exact top-level folder.
- The dataset files are ignored by Git on purpose and should not be committed to GitHub.
