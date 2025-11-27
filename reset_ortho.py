import rasterio
from rasterio.transform import Affine

INPUT = "/tmp/Castello_Neuschwanstein/images/Castello_Neuschwanstein_s_0_75.tif"

# Apri il file GeoTIFF
with rasterio.open(INPUT) as src:
    meta = src.meta.copy()
    data = src.read()  # legge tutti i canali

    # Calcola lo shift: sposta il centro dell'immagine in (0,0)
    # centro originale
    cx, cy = src.transform * (src.width / 2, src.height / 2)

    # nuova trasformazione: trasla tutto in modo che cx,cy diventi 0,0
    new_transform = src.transform * Affine.translation(-cx+100, -cy+200)

    meta.update(transform=new_transform)

    # salva il nuovo GeoTIFF
    with rasterio.open("Castello_Neuschwanstein_s_0_75.tif", "w", **meta) as dst:
        dst.write(data)