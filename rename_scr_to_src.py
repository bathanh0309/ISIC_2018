import json
import os

# 1. Update main.ipynb
nb_path = 'd:/ISIC2018/main.ipynb'
if os.path.exists(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb_str = f.read()
    
    # Replace all occurrences of "scr" with "src" cases where it's likely part of paths or imports
    # Specifically looking for "from scr.", "from scr ", "import scr.", "scr/", "'scr'"
    nb_str = nb_str.replace('from scr.', 'from src.')
    nb_str = nb_str.replace('/scr/', '/src/')
    nb_str = nb_str.replace('\"scr/\"', '\"src/\"')
    nb_str = nb_str.replace('\'scr\'', '\'src\'')
    nb_str = nb_str.replace('scr_path', 'src_path') # logic names
    
    # Simple replacement for "scr" in strings if it looks like a directory
    nb_str = nb_str.replace('\"scr\"', '\"src\"')
    nb_str = nb_str.replace('\'scr\'', '\'src\'')
    # More generally, replace "scr" if it's a standalone folder name in a path
    nb_str = nb_str.replace('/scr', '/src')
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        f.write(nb_str)
    print("Updated main.ipynb")

# 2. Update README.md
readme_path = 'd:/ISIC2018/README.md'
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('scr/', 'src/')
    content = content.replace('scr ', 'src ')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Updated README.md")

# 3. Check for any other files in the root that might mention scr
for filename in os.listdir('d:/ISIC2018'):
    if filename.endswith('.py') and filename != 'rename_scr_to_src.py':
        file_path = os.path.join('d:/ISIC2018', filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            c = f.read()
        if 'scr' in c:
            c = c.replace('from scr.', 'from src.')
            c = c.replace('import scr', 'import src')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(c)
            print(f"Updated {filename}")
