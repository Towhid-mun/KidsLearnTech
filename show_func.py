from pathlib import Path
text = Path(r"C:\Working\ChildEdu\main.py").read_text()
import re
match = re.search(r"def _render_cartoon_frame[\s\S]+?return np.array\(image\)\n", text)
print(match.group() if match else 'not found')
