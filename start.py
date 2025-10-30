import os
import subprocess

def process_files(input_dir):
    """
    Legge tutti i file nella cartella 'input_dir', tralascia 'tmp' e i suoi contenuti,
    ed esegue lo script /app/python.py su ciascun file.
    """
    for root, dirs, files in os.walk(input_dir):
        # Salta la cartella 'tmp' e il suo contenuto
        if 'tmp' in dirs:
            dirs.remove('tmp')
        
        if 'tmp_1' in dirs:
            dirs.remove('tmp_1')

        for filename in files:
            file_path = os.path.join(root, filename)

            # Puoi aggiungere un filtro per tipo di file, ad esempio solo .obj
            if not filename.lower().endswith('.glb'):
                continue
            
            print (f"####################################################################")
            print (f"####################################################################")
            print (f"##### Elaborazione del file: {file_path}")
            print (f"####################################################################")

            lat, lon = None, None
            if (filename == "piazzaDuomoTrento.glb"):
                lat, lon = "46.066900", "11.121600"
            elif (filename == "sagradaFamilia.glb"):
                lat, lon = "41.403620", "2.174350"
            elif (filename == "torreEiffel.glb"):
                lat, lon = "48.858237", "2.294481"

            # Costruisci il comando
            cmd = [
                "python3",
                "/app/main.py",
                "-i", file_path,
                "-o", "/data/output/test",
                "--geoloc_model", "geoclip",
                "--lat", lat,
                "--lon", lon
            ]

            # Esegui il comando
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Errore durante l'elaborazione di {filename}: {e}")
            


if __name__ == "__main__":
    INPUT_DIR = "/data/input/"
    GEOLOC_MODEL = "gemini"

    process_files(INPUT_DIR)
