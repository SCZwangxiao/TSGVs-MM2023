# Data Preparation

We provide some tips for MMAction2 data preparation in this file.

<!-- TOC -->

- [Data Directory Structures](#data-directory-structures)
- [Getting Data](#getting-data)
  - [ActivityNet Captions](#ActivityNet-Captions)
  - [TACoS](#TACoS)
  - [MAD](#MAD)

<!-- TOC -->

## Data Directory Structures

```
./data
    /<dataset_name_1>
        /feature
            *.hdf5
            ...
        train.json
        val.json
        test.json
    ...
```

## Getting Data

Please follow the guide below to prepare the datasets.

### ActivityNet Captions
- Please [download](https://pan.baidu.com/s/1A5SkGBZwVvI_55Zlm2QC4w?pwd=a6dg) the annotations and pre-extracted features.
- Put all files into ``./data`` of the repository.

### TACoS
- Please [download](https://pan.baidu.com/s/13HIkoO5hfuIya6_77qmJPg?pwd=iech) the annotations and pre-extracted features.
- Put all files into ``./data`` of the repository.
#### MAD
- Follow the instructions in [MAD repository](https://github.com/Soldelli/MAD) to download the dataset.
- Using the script ``tools/data/mad/convertion.py`` to convert the data into our standard format.
