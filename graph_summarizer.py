import os
import argparse
from datetime import datetime
import subprocess
import sys
from collections import Counter

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pokročilé spouštění table_summarizer.py s číselným řazením")
    parser.add_argument("--version", type=str, required=True, help="Verze pro výstupní název (např. 2.0)")
    parser.add_argument("--datetime", type=str, required=True, 
                        help="Filtr data a času ve formátu 'YYYY-MM-DD HH:MM:SS'")
    return parser.parse_args()

def numerical_sort_key(folder_name, param_index):
    """
    Vrátí klíč pro řazení na základě číselné hodnoty parametru na daném indexu.
    Převede např. '0-15' na float 0.15 pro správné matematické seřazení.
    """
    try:
        parts = folder_name.split("_")
        # parts je 'pfns', parametry začínají od indexu 1
        val_str = parts[param_index + 1]
        # Nahradí pomlčku tečkou (0-15 -> 0.15) pro převod na float
        return float(val_str.replace("-", "."))
    except (IndexError, ValueError):
        # Záložní řešení, pokud by parsování čísla selhalo
        return folder_name

def main():
    args = parse_arguments()
    
    try:
        filter_dt = datetime.strptime(args.datetime, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print("Chyba: Formát data a času musí být 'YYYY-MM-DD HH:MM:SS'")
        sys.exit(1)
        
    base_dir = "coco_plots"
    if not os.path.exists(base_dir):
        print(f"Chyba: Složka '{base_dir}' neexistuje.")
        sys.exit(1)

    # 1. Načtení a filtrace složek podle času změny (mtime)
    valid_folders = []
    for entry in os.scandir(base_dir):
        if entry.is_dir() and entry.name.startswith("pfns_"):
            mtime = datetime.fromtimestamp(entry.stat().st_mtime)
            if mtime > filter_dt:
                valid_folders.append(entry.name)

    if not valid_folders:
        print(f"Nenalezeny žádné složky v '{base_dir}' mladší než {filter_dt}")
        sys.exit(0)

    print(f"Nalezeno {len(valid_folders)} složek splňujících časový filtr.")

    # Definice názvů pozic parametrů
    param_names = ["ts", "ps", "rf", "tf", "uf", "tsr"]

    # 2. Naparsování složek do strukturovaných dat
    parsed_data = []
    for folder in valid_folders:
        parts = folder.split("_")
        tokens = parts[1:]
        
        folder_dict = {"_full_name": folder}
        for i, name in enumerate(param_names):
            if i < len(tokens):
                folder_dict[name] = tokens[i]
            else:
                folder_dict[name] = None
        parsed_data.append(folder_dict)

    # 3. Zjištění, které parametry mají více než jednu unikátní hodnotu
    variable_params = []
    for name in param_names:
        unique_values = {d[name] for d in parsed_data if d[name] is not None}
        if len(unique_values) > 1:
            variable_params.append(name)

    if not variable_params:
        print("Žádný z parametrů nemá variabilní hodnoty napříč složkami.")
        sys.exit(0)

    print(f"Detekované proměnlivé parametry: {variable_params}")

    # 4. Pro každý proměnlivý parametr najdeme složky, seřadíme je a spustíme skript
    for target_param in variable_params:
        print(f"\n=== Zpracování parametru: {target_param} ===")
        
        # Index cílového parametru (0 pro ts, 1 pro ps, atd.) pro potřeby řazení
        target_index = param_names.index(target_param)
        
        # Ostatní parametry, které chceme zafixovat
        other_params = [p for p in param_names if p != target_param]
        
        # Najdeme nejčastější kombinaci hodnot pro ostatní parametry
        combinations = []
        for d in parsed_data:
            combo = tuple(d[p] for p in other_params)
            combinations.append(combo)
            
        # OCHRANA: Pokud z nějakého důvodu nejsou žádné kombinace, parametr přeskočíme
        if not combinations:
            print(f"Varování: Pro parametr {target_param} nebyly nalezeny žádné kombinace parametrů.")
            continue
            
        # Bezpečné získání nejčastější kombinace
        most_common_list = Counter(combinations).most_common(1)
        if not most_common_list:
            print(f"Varování: Nepodařilo se určit nejčastější kombinaci pro {target_param}.")
            continue
            
        most_common_combo = most_common_list[0][0]
        fixed_values = dict(zip(other_params, most_common_combo))
        
        # Vyfiltrujeme složky odpovídající nejčastější kombinaci
        selected_folders = []
        for d in parsed_data:
            match = True
            for p in other_params:
                if d[p] != fixed_values[p]:
                    match = False
                    break
            if match:
                selected_folders.append(d["_full_name"])
                
        if not selected_folders:
            print(f"Pro parametr {target_param} nebyly vybrány žádné složky.")
            continue

        # 5. NUMERICKÉ ŘAZENÍ SLOŽEK podle hodnoty cílového parametru
        selected_folders.sort(key=lambda f: numerical_sort_key(f, target_index))

        # Sestavení výstupního názvu obrázku
        output_filename = f"coco_plots/result_table-{target_param}-{args.version}.png"
        
        # Sestavení finálního příkazu pro subprocess (používá aktuální venv)
        cmd = [
            sys.executable, "table_summarizer.py",
            "--folders"
        ] + selected_folders + [
            "--start_f", "15",
            "--end_f", "19",
            "--output", output_filename
        ]
        
        print(f"Vybrané a numericky seřazené složky pro parametr '{target_param}':")
        for f in selected_folders:
            print(f"  - {f}")
        print(f"Spouštím příkaz s výstupem do: {output_filename}")
        
        # Spuštění table_summarizer.py
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print(f"Chyba: Skript table_summarizer.py selhal pro parametr {target_param} s kódem {result.returncode}")
        else:
            print(f"Úspěšně dokončeno pro parametr {target_param}")

if __name__ == "__main__":
    main()
