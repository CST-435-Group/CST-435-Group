import kaggle
kaggle.api.dataset_download_files('junaid6731/hospital-reviews-dataset', 
                                   path='./data', unzip=True)

print("Dataset downloaded and extracted successfully.")
