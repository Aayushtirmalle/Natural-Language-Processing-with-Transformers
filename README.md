# QA-Robot-Med-INLPT-WS2023
Question Answering (QA) system tailored for the medical domain on <em>Intelligence</em>

## Table of Contents

- [Data](#data)

## data

- Infos about our dataset
    - data source: [pubmed](https://pubmed.ncbi.nlm.nih.gov/)
    - data format:
      - PMID
      - Title
      - Author
      - Journal Title
      - Publication Date
      - Abstract
    - data size: there are 20 articles in a txt file, so we have 20 * 44 = 880 articles.
- Example
![dataset example](assets/images/dataset_example.jpg)

- Download

```bash
# run the script /data_process/load_data.py to download more data
pip install Bio
python /data_process/load_data.py
```