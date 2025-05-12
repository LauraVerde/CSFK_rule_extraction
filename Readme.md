# Extracting Knowledge from Data in Lightweight Digital Twin Construction


## 📦 Table of Contents

- [Overview](#-overview)  
- [Installation](#️-installation)  
- [Dataset](#-dataset)  
- [Running the Code](#️-running-the-code)  
- [Reproducibility](#-reproducibility)  
- [License](#-license)  
- [Contact](#-contact)

---

## 🧠 Overview

This repository contains the replication materials for the paper **"Extracting Knowledge from Data in
Lightweight Digital Twin Construction"**, which investigates the extraction of prediction rules of kidney damage in
patients with Congenital Solitary Functioning Kidney (CSFK). The work has the objective to create a lightweight model learning rules that can be interpretable by
medical staff. To this aim, an ad-hoc dataset is used collecting different features of the abovementioned patients.

This replication package includes:
- Scripts for preprocessing the CSFK dataset  
- Implementation of Random Forest training and evaluation  
- Extraction of the rules learnt by the RF model

---

## ⚙️ Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/LauraVerde/CSFK_rule_extraction.git
cd CSFK_rule_extraction
pip install -r requirements.txt
```

---

## 📁 Dataset

The dataset used in this repository is not publicly available. To obtain the dataset you must contact me. 
The project contains an exceprt of the dataset.

---

## ▶️ Running the Code

```bash
python main.py _dataset_name_
```

As a result, the folder of the project contains all the intermediate datasets, the report in textual form and the images
graphically describes the learnt model.

---


## ▶️ Reproducibility

The code here reported is enough to replicate the graphs reported in the paper. This notwithstanding, the exact results could
be generated also with the presence of original dataset.

---

## 📝 License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

See the [LICENSE](./LICENSE-2.0.txt) file for more details.

---

## 📬 Contact

For questions, comments, or collaboration inquiries, please contact:

- **Laura Verde**  
- 📧 laura.verde@unicampania.it  
- 🏢 [Istitutional Site](https://www.matfis.unicampania.it/dipartimento/docenti-csa?MATRICOLA=711049)  
- 🌐 [ORCID](https://orcid.org/0000-0003-2422-1732)

Other contributors are [Giusy D'Angelo](giusy.dangelo3@studenti.unicampania.it) and [Roberta Petruolo](roberta.petruolo@studenti.unicampania.it)
