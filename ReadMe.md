## Advaith Malladi

### Directory Structure

```
.
├── calc_bleu.py
├── decoder
│   ├── modules.py
│   └── __pycache__
│       ├── modules.cpython-311.pyc
│       └── modules.cpython-39.pyc
├── decoder_model.pt
├── dev.csv
├── dev.en
├── dev.fr
├── encoder
│   ├── modules.py
│   └── __pycache__
│       ├── modules.cpython-310.pyc
│       ├── modules.cpython-311.pyc
│       └── modules.cpython-39.pyc
├── encoder_model.pt
├── eng_idx2wrd.pt
├── eng_wrd2idx.pt
├── fr_idx2wrd.pt
├── fr_wrd2idx.pt
├── ReadMe.md
├── test.csv
├── test.en
├── test.fr
├── train.csv
├── train.en
├── train.fr
└── train.py

4 directories, 26 files

```
### Download:
```
train.en, train.fr dev.en, dev.fr, test.en, test.fr dev.csv, test.csv, train.csv eng_wrd2idx.pt, eng_idx2wrd.pt, fr_wrd2idx.pt, fr_idx2wrd.pt encoder_model.pt, decoder_model.pt
```
### from the the following link, please follow the directory structure: https://iiitaphyd-my.sharepoint.com/:f:/g/personal/advaith_malladi_research_iiit_ac_in/EgXvSyD6LL5Drq2PQBtfMG0BuhMos_iUmKbSfGPPlkYdog?e=LurvLD


### To look at the code for the encoding layers implemented by me, open:
```
cd encoder
modules.py
```
### To look at the code for the decoding layers implemented by me, open:
```
cd decoder
modules.py
```

### TO train the end to end translation model using the best hyperparameters, run:
```
python3 train.py
```

### TO calculate the BLEU score, run:
```
python3 calc_bleu.py
```
