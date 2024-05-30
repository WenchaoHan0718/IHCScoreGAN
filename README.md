# CPAI
Code base for CPAI work

## Training

To train the model, you must populate the `trainA` and `trainB` folders with source (IHC) and target (synthetic) datasets. Your folder structure should look like:
```
└── dataset
    └── trainA
        └── name_of_trainA_img.png
            ...
    └── trainB
        └── json
            └── name_of_trainB_file.json
            ...
        └── overlay
            └── name_of_trainB_img.png
            ...
├── .gitignore
├── IHCScoreGAN.py
├── README.md
├── dataset.py
├── main.py
├── modules.py
└── utils.py
```

You can then run the model from the `main.py` file or through your terminal window, like so:
```
python main.py --exp_name bcdataset --phase train --num_workers 8 --batch_size 4 --input_dir dataset --resume False
```

## Testing

To test the model, you must populate the `testA` folder with source (IHC) datasets. Your folder structure should look like:
```
└── dataset
    └── testA
        └── name_of_testA_img.png
            ...
└── results
    └── <experiment name>
        └── model
            └── model_weights_file.pt
├── .gitignore
├── IHCScoreGAN.py
├── README.md
├── dataset.py
├── main.py
├── modules.py
└── utils.py
```

You can then run the model from the `main.py` file or through your terminal window, like so:
```
python main.py --exp_name bcdataset --phase test --num_workers 8 --batch_size 4 --load_path results/<experiment name>/model/model_weights_file.pt --results_dir results --save_images True
```