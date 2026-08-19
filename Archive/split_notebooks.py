import json
import os

cwd = 'w:/OneDrive - National Economics University/ELTE/Synthetic Data Gen/New experiments'
files = [
    'run_experiments_Plant_oil.ipynb',
    'run_experiments_brewed_vinegar.ipynb',
    'run_experiments_chinese_wine.ipynb',
    'run_experiments_coffee.ipynb',
    'run_experiments_wine_spoilage.ipynb'
]

for file in files:
    file_path = os.path.join(cwd, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Original nb has 6 cells: 1 markdown, 5 code (for seeds 1, 2, 5, 10, 20)
    # Part 1: seeds 1, 2, 5
    # Part 2: seeds 10, 20
    
    nb1 = json.loads(json.dumps(nb))
    nb2 = json.loads(json.dumps(nb))
    
    # Update markdown cell
    nb1['cells'][0]['source'] = [
        "# Run Experiment Pipeline\n",
        "The experiment is split into cells corresponding to Seed Size: 1, 2, 5.\n"
    ]
    nb2['cells'][0]['source'] = [
        "# Run Experiment Pipeline\n",
        "The experiment is split into cells corresponding to Seed Size: 10, 20.\n"
    ]
    
    # Filter cells
    cells1 = [nb1['cells'][0]]
    cells2 = [nb2['cells'][0]]
    
    for cell in nb['cells'][1:]: # skip markdown
        source = "".join(cell['source'])
        if "--seed 1\n" in source or "--seed 2\n" in source or "--seed 5\n" in source or "--seed 1" in source or "--seed 2" in source or "--seed 5" in source:
            if "--seed 10" not in source and "--seed 20" not in source:
                cells1.append(cell)
        if "--seed 10\n" in source or "--seed 20\n" in source or "--seed 10" in source or "--seed 20" in source:
            cells2.append(cell)
            
    nb1['cells'] = cells1
    nb2['cells'] = cells2
    
    # Save back original file as part 1
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb1, f, indent=1)
        
    # Save part 2
    name, ext = os.path.splitext(file_path)
    part2_path = f"{name}_part2{ext}"
    with open(part2_path, 'w', encoding='utf-8') as f:
        json.dump(nb2, f, indent=1)

print("Split completed.")
