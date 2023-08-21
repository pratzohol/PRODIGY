import pandas as pd
import os

rel_path = "./arxiv_CS_categories.txt"


f = open(os.path.join(os.path.dirname(__file__), rel_path), "r").readlines() # get the pwd() and append rel_path. Then open the file and read the lines

state = 0
result = {
    "id": [],
    "name": [],
    "description": []
}

for line in f:
    if state == 0:
        assert line.strip().startswith("cs.")
        category = "arxiv " + " ".join(line.strip().split(" ")[0].split(".")).lower() # e. g. arxiv cs ai
        # 7th index to 2nd last char, since last char is \n
        name = line.strip()[7:-1]  # e. g. Artificial Intelligence
        result["id"].append(category)
        result["name"].append(name)
        state = 1
        continue
    elif state == 1:
        description = line.strip()
        result["description"].append(description)
        state = 2
        continue
    elif state == 2:
        state = 0
        continue

arxiv_cs_taxonomy = pd.DataFrame(result)