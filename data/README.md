# Data folder

This folder is **not** committed to Git (see `.gitignore`). Put the following here:

```
data/
├── vpf_artifacts/            required: engine files from the evaluation notebook (Part 5)
│   ├── catalog_embeddings.npy
│   ├── products.csv
│   ├── model_config.json
│   ├── search_settings.json
│   └── manifest.json
└── images/                   optional: catalog images, only needed to show product pictures
    ├── 10000.jpg
    └── ...
```

* `vpf_artifacts/`: download it from the **Output** tab of the Kaggle notebook *VPF 05 · Evaluation*.
* `images/`: the `images` folder of the Kaggle *Fashion Product Images* dataset. Do not redistribute these
  images (check the dataset license). Instead of copying them you can point to them with `VPF_IMAGE_DIR`.
