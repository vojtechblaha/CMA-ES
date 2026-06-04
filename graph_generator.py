import os
import re
import subprocess
import sys

# Nastavení cest
ZAKLADNI_SLOZKA = "exdata-meta"

if not os.path.exists(ZAKLADNI_SLOZKA):
    print(f"Chyba: Složka '{ZAKLADNI_SLOZKA}' neexistuje!")
    sys.exit(1)

# 1. Získání jedinečných názvů experimentů
experimenty = set()

for polozka in os.listdir(ZAKLADNI_SLOZKA):
    cesta = os.path.join(ZAKLADNI_SLOZKA, polozka)
    
    # Kontrola, zda jde o složku a začíná správným prefixem
    if os.path.isdir(cesta) and polozka.startswith("pfns_"):
        # Odstraní "_bbob_" a vše, co následuje za tím
        cisty_nazev = re.sub(r"_bbob_.*$", "", polozka)
        experimenty.add(cisty_nazev)

# Převod na seřazený seznam
seznam_experimentu = sorted(list(experimenty))

if not seznam_experimentu:
    print("Nebyly nalezeny žádné experimenty odpovídající masce.")
    sys.exit(0)

print(f"Nalezeno {len(seznam_experimentu)} unikátních experimentů.")
print(f"Používám Python z venv: {sys.executable}")

# 2. Spouštění příkazů v cyklu
for exp in seznam_experimentu:
    # sys.executable zajistí, že se použije přesně ten python, ve kterém běží tento skript (váš venv)
    dim = 5
    budget = 10000
    if "_dim_" in exp:
        dim = int(re.search(r"_dim_(\d+)", exp).group(1))
        budget = 2000 * dim
    prikaz = [
        sys.executable, "coco_eval_graph.py", "exdata-meta", str(dim), "15", "19", exp,
        "--ref-tags", "2009", "2010", "2012", "2013", "2014-others", "2015-CEC", 
        "2015-GECCO", "2016", "2017", "2017-others", "2018", "2018-others", 
        "2019", "2020", "2021", "2022", "2023", 
        "--cache-dir", "coco_cache", 
        "--budget", str(budget)
    ]
    
    print(f"\n---> Spouštím evaluaci pro: {exp}")
    
    # Spuštění příkazu (vypisuje výstup v reálném čase do konzole)
    vysledek = subprocess.run(prikaz, text=True)
    
    if vysledek.returncode != 0:
        print(f"Varování: Příkaz pro {exp} skončil s chybou (kód {vysledek.returncode}).")

print("\nVšechny evaluace byly dokončeny.")
