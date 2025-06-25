import geopandas as gpd
import fiona
import shapely

print("GeoPandas version:", gpd.__version__)
print("Fiona version:", fiona.__version__)
print("Shapely version:", shapely.__version__)

# Optional: check if PyGEOS is available
try:
    import pygeos
    print("PyGEOS version:", pygeos.__version__)
except ImportError:
    print("PyGEOS not installed")
