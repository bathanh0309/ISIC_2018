import json

nb_path = 'd:/ISIC2018/main.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 4 to include TTA flag in config dict
# Update Cell 6 to use TTA flag during test evaluation

found_4 = False
found_6 = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_text = "".join(cell['source'])
        
        # Update config dict in Cell 4
        if "# Cell 4: Training Stage" in source_text:
            if "'USE_TTA_VALIDATION': config.USE_TTA_VALIDATION" not in source_text:
                new_source = []
                for line in cell['source']:
                    new_source.append(line)
                    if "'idx2label': idx2label" in line:
                        # Add comma to previous line if missing (JSON/Dict style)
                        if not line.strip().endswith(','):
                            new_source[-1] = line.replace('idx2label', 'idx2label,')
                        new_source.append("    'USE_TTA_VALIDATION': config.USE_TTA_VALIDATION\n")
                cell['source'] = new_source
                found_4 = True

        # Update evaluation in Cell 6
        if "# Cell 6: Đánh giá trên Test Set" in source_text:
            if "use_tta=config.USE_TTA_TEST" not in source_text:
                new_source = []
                for line in cell['source']:
                    if "model, test_loader, criterion, DEVICE" in line:
                        new_source.append(line.replace("DEVICE", "DEVICE, use_tta=config.USE_TTA_TEST"))
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                found_6 = True

if found_4 or found_6:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print("Notebook updated with TTA support in training and evaluation.")
else:
    print("Could not find cells to update.")
