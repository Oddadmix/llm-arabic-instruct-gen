from datasets import load_dataset, Dataset
import os
import json

folder = "my_output_text_gemma_2_9b_try4"


## loop over all files in the folder

data = []

## for all files in the folder
for file in os.listdir(folder):
    if file.endswith(".json"):
        with open(os.path.join(folder, file), "r") as f:
            content = f.read()

            content = json.loads(content)
            data.append({
                "messages": [
                {
                    "content": content["question"],
                    "role"  : "user"
                },
                {
                    "content": content["answer"],
                    "role"  : "assistant"
                }
            ]
            })

# Create a Dataset from the data
dataset = Dataset.from_list(data)

dataset.push_to_hub("oddadmix/recent-gemma-9b-qa", private=True)