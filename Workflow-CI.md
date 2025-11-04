# Kriteria 3 - Workflow CI (Continuous Integration)

repository:
https://github.com/NauvalGymnasti/bankruptcy-analysis-system-Workflow-CI.git

## Deskripsi
Tahap ini mengotomatisasi pelatihan model menggunakan GitHub Actions dan MLflow Project.
Workflow CI akan menjalankan proses retraining model setiap kali ada perubahan (trigger push) di repositori.

## Struktur Folder
Workflow-CI/
├── .github/workflows/
│   └── ci.yml
├── MLProject/
│   ├── MLProject
│   ├── conda.yaml
│   ├── modelling.py
│   ├── Company_Bankruptcy_Prediction_data_preprocessing/Preprocessing.py
│   └── dataset/Company_Bankruptcy_Prediction_data_preprocessing.csv
└── README.txt

## Workflow Trigger
- Trigger otomatis saat ada commit atau pull request ke branch `main`.

## Output
- Model retrained otomatis melalui CI.
- Artefak model disimpan di repositori atau storage yang ditentukan.

## Tools
- MLflow Project
- GitHub Actions
- Conda Environment
