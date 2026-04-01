from pathlib import Path


def resolve_txt_path(path) -> Path:
    file_path = Path(path)
    if file_path.suffix != ".txt":
        file_path = file_path.with_suffix(".txt")
    return file_path

def remove_confidence(lines): #remove last column with confidence 
    new_lines = []
    for line in lines:
        parts = [part.strip() for part in line.strip().split(",") if part.strip() != ""]
        if not parts:
            continue
        if len(parts) > 1:
            parts = parts[:-1]
        new_lines.append(",".join(parts) + "\n")
    return new_lines


def label(path):
    file_path = resolve_txt_path(path)
    
    with file_path.open("r", encoding="utf-8") as f: #read file
        lines = f.readlines()

    new_lines = remove_confidence(lines) #Pseudo labeling 
    
    with file_path.open("w", encoding="utf-8") as f: #save new file
        f.writelines(new_lines)
