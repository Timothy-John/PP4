from utils import perform_training

datasets = ["CoversIndian"]
models = ["melspectrogram", "24khz", "48khz"]

for dataset in datasets:
    for model in models:
        print("Processing", dataset, "with model", model, "...")
        perform_training(dataset, model)
