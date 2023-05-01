import argparse
import torch
import torch.nn.functional as F
import os
import pandas as pd
from models.resnet import ResNet
from utils._dataset import read_data, create_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str.lower, required=True)
    args = parser.parse_args()
    
    # Validating the passed arguments
    assert os.path.exists(args.input_dir), "Please enter a valid input directory."
    assert (
        args.model_name == "resnet"
    ), f"Please enter a valid model name."
    os.makedirs(args.output_dir, exist_ok=True)

    # Reading the data
    df = read_data(
        input_dir=f"{args.input_dir}/test1",
        inference_mode=True
    )
    
    X = df["data"].values.tolist()
    y = df["label"].values.tolist()
    
    # Creating the dataloader
    dataloader = create_dataloader(
        X=X,
        y=y,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )
    
    # Creating the model
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    
    model = ResNet().to(device)
    model.load_state_dict(
        torch.load(f"{args.output_dir}/{args.model_name}.pth")["model_state_dict"]
    )
    model.eval()
    
    # Creating the submission dataframe
    submission = pd.DataFrame()
    
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            name = int(df["name"][i])
            data, _ = batch[0].to(device), batch[1]

            output = model(data)
            prediction = F.softmax(output, dim=-1)
            prediction = int(prediction.argmax(dim=1, keepdim=True).squeeze(-1).item())
            
            row = pd.DataFrame({
                "id": [name],
                "label": [prediction]
            })
            submission = pd.concat([submission, row], axis=0)
    
    submission = submission.reset_index(drop=True)
    submission = submission.sort_values(by="id", ascending=True)
    submission.to_csv(f"{args.output_dir}/submission.csv", index=False, sep=";")