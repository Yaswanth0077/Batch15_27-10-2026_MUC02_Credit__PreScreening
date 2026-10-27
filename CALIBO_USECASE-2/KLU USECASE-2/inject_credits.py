import json, uuid

proj = r'e:\CaliboAiAcad\MCU2\CALIBO_USECASE-2\KLU USECASE-2'

# Step credits to inject after each major step heading
STEP_CREDITS = {
    "Step 5.1": "👤 **Built by: Velagapudi Haritha (2300030248)** — Data Analyst",
    "Step 5.2": "👤 **Built by: Velagapudi Haritha (2300030248)** — Data Analyst",
    "Step 5.3": "👤 **Built by: Kunapuli Bala Sri Sarvani (2300030957)** — ML Engineer",
    "Step 5.4": "👤 **Built by: Kunapuli Bala Sri Sarvani (2300030957)** — ML Engineer",
    "Step 5.5": "👤 **Built by: Kunapuli Bala Sri Sarvani (2300030957)** — ML Engineer",
    "Step 5.6": "👤 **Built by: Kunapuli Bala Sri Sarvani (2300030957)** — ML Engineer",
    "Step 5.7": "👤 **Built by: Yaswanth Kumar Reddy Vogireddy (2300031126)** — Project Lead",
    "C1.":      "👤 **Built by: Yaswanth Kumar Reddy Vogireddy (2300031126)** — Project Lead",
    "C2.":      "👤 **Built by: Yaswanth Kumar Reddy Vogireddy (2300031126)** — Project Lead",
    "C3.":      "👤 **Built by: Yaswanth Kumar Reddy Vogireddy (2300031126)** — Project Lead",
    "C4.":      "👤 **Built by: Yaswanth Kumar Reddy Vogireddy (2300031126)** — Project Lead",
    "C5.":      "👤 **Built by: Yaswanth Kumar Reddy Vogireddy (2300031126)** — Project Lead",
    "C6.":      "👤 **Built by: Yaswanth Kumar Reddy Vogireddy (2300031126)** — Project Lead",
}

def inject_credits(nb):
    new_cells = []
    for cell in nb['cells']:
        new_cells.append(cell)
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell.get('source', []))
            for key, credit in STEP_CREDITS.items():
                if f'## {key}' in src or f'# {key}' in src:
                    # Only add if next cell isn't already a credit cell
                    credit_cell = {
                        "cell_type": "markdown",
                        "id": str(uuid.uuid4())[:8],
                        "metadata": {},
                        "source": [f"> {credit}\n"]
                    }
                    new_cells.append(credit_cell)
                    break
    return new_cells

# Update both notebooks (Notebook + Master)
for fname in ["YASWANTH_MUC02_KLU_Notebook.ipynb", "YASWANTH_MUC02_KLU_Master.ipynb"]:
    fp = proj + '\\' + fname
    with open(fp, encoding='utf-8') as f:
        nb = json.load(f)
    nb['cells'] = inject_credits(nb)
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Credits injected: {fname} ({len(nb['cells'])} cells)")
