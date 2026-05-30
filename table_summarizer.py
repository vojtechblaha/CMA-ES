import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from glob import glob

def create_plot_grid():
    # Nastavení argumentů příkazové řádky
    parser = argparse.ArgumentParser(description="Spojí asymetricky oříznuté grafy s popisky do tabulky.")
    parser.add_argument("--folders", nargs="+", required=True, 
                        help="Seznam názvů podsložek bez prefixu (např. ts_200 rf_0_5)")
    parser.add_argument("--start_f", type=int, required=True, help="Číslo první funkce (např. 15)")
    parser.add_argument("--end_f", type=int, required=True, help="Číslo poslední funkce (např. 24)")
    parser.add_argument("--output", type=str, default="vysledna_tabulka.png", help="Název výstupního souboru")
    
    args = parser.parse_args()
    
    base_dir = "coco_plots"
    functions = list(range(args.start_f, args.end_f + 1))
    
    # Definice asymetrického ořezu podle zadání
    CROP_TOP = 64
    CROP_BOTTOM = 104
    CROP_LEFT = 117
    CROP_RIGHT = 24
    
    # Nastavení velikosti postranních panelů pro popisky (v pixelech)
    LABEL_COLUMN_WIDTH = 600  # Šířka levého sloupce pro názvy složek
    LABEL_ROW_HEIGHT = 60     # Výška horního řádku pro čísla funkcí
    FONT_SIZE = 40 
    
    # 1. Krok: Zjištění rozměrů jednoho oříznutého obrázku
    first_folder = f"{args.folders[0]}"

    matches = sorted(glob(os.path.join(base_dir, first_folder, f"bbob_f{args.start_f}_dim*_ecdf.png")))
    if matches:
        first_img_path = matches[0]
    else:
        first_img_path = None
    
    if not os.path.exists(first_img_path):
        print(f"Chyba: Vzorový obrázek neexistuje: {first_img_path}")
        return

    with Image.open(first_img_path) as img:
        orig_w, orig_h = img.size
        # Výpočet nových rozměrů po asymetrickém oříznutí
        tile_w = orig_w - CROP_LEFT - CROP_RIGHT
        tile_h = orig_h - CROP_TOP - CROP_BOTTOM
        
    if tile_w <= 0 or tile_h <= 0:
        print("Chyba: Hodnoty ořezu jsou větší než samotný obrázek.")
        return

    # 2. Krok: Výpočet celkových rozměrů plátna včetně popisků
    grid_w = LABEL_COLUMN_WIDTH + (len(functions) * tile_w)
    grid_h = LABEL_ROW_HEIGHT + (len(args.folders) * tile_h)
    
    # Vytvoření bílého plátna (pozadí pod textem)
    result_image = Image.new("RGBA", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(result_image)
    
    # Načtení výchozího písma
    try:
        # Pokus o načtení standardních systémových fontů podle OS
        if os.name == 'nt':  # Windows
            font = ImageFont.truetype("arial.ttf", FONT_SIZE)
        elif os.path.isdir('/Library/Fonts'):  # macOS
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf", FONT_SIZE)
        else:  # Linux / ostatní
            font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except Exception:
        print("Varování: Nepodařilo se načíst systémový font, zvětšuji výchozí.")
        # Záložní řešení, pokud systémový font selže
        font = ImageFont.load_default(size=FONT_SIZE)

    # 3. Krok: Vykreslení horního řádku s čísly funkcí (osa X)
    for x_idx, f_num in enumerate(functions):
        text = f"f{f_num}"
        # Výpočet středu buňky pro zarovnání textu
        cell_left = LABEL_COLUMN_WIDTH + (x_idx * tile_w)
        cell_right = cell_left + tile_w
        
        # Získání velikosti textu pro přesné vycentrování
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        text_x = cell_left + (tile_w - text_w) // 2
        text_y = (LABEL_ROW_HEIGHT - text_h) // 2
        
        draw.text((text_x, text_y), text, fill="black", font=font)

    # 4. Krok: Průchod složkami, vkládání obrázků a levých popisků (osa Y)
    for y_idx, folder_suffix in enumerate(args.folders):
        folder_name = f"{folder_suffix}"
        folder_path = os.path.join(base_dir, folder_name)
        
        # Vykreslení textu složky do levého sloupce
        cell_top = LABEL_ROW_HEIGHT + (y_idx * tile_h)
        cell_bottom = cell_top + tile_h
        
        bbox = draw.textbbox((0, 0), folder_suffix, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Odsazení 10px od levého okraje pro zarovnání doleva, vertikálně na střed
        text_x = 10 
        text_y = cell_top + (tile_h - text_h) // 2
        
        draw.text((text_x, text_y), folder_suffix, fill="black", font=font)
        
        # Vkládání jednotlivých oříznutých grafů
        for x_idx, f_num in enumerate(functions):
            matches = sorted(glob(os.path.join(folder_path, f"bbob_f{f_num}_dim*_ecdf.png")))
            if matches:
                img_path = matches[0]
            else:
                img_path = None
            
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    # Definice výřezu: (vlevo, nahoře, vpravo, dole)
                    box = (CROP_LEFT, CROP_TOP, orig_w - CROP_RIGHT, orig_h - CROP_BOTTOM)
                    cropped_img = img.crop(box)
                    
                    # Výpočet pozice na plátnu s ohledem na postranní panely
                    pos_x = LABEL_COLUMN_WIDTH + (x_idx * tile_w)
                    pos_y = LABEL_ROW_HEIGHT + (y_idx * tile_h)
                    
                    result_image.paste(cropped_img, (pos_x, pos_y))
            else:
                print(f"Varování: Soubor {img_path} nebyl nalezen. Pozice zůstane prázdná.")

    # 5. Krok: Uložení výsledku
    result_image.save(args.output)
    print(f"Hotovo! Tabulka byla uložena jako '{args.output}' ({grid_w}x{grid_h}px).")

if __name__ == "__main__":
    create_plot_grid()
