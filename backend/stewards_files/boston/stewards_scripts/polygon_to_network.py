import argparse
import time
import os
import geopandas as gpd

from tile2net.raster.project import Project
from tile2net.raster.raster import Raster
from tile2net.raster.pednet import PedNet


def main():

    parser = argparse.ArgumentParser(
        description="Run Tile2Net topology engine on external polygon shapefile."
    )

    parser.add_argument("--input", required=True, help="Path to polygon shapefile (.shp)")
    parser.add_argument("--output", required=True, help="Output shapefile name (without .shp)")
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                        help="Bounding box: min_lon min_lat max_lon max_lat")

    args = parser.parse_args()

    start = time.time()

    print("\nLoading polygons...")
    gdf = gpd.read_file(args.input)

    print("Polygon count:", len(gdf))

    # Ensure required column exists
    if "f_type" not in gdf.columns:
        gdf["f_type"] = "sidewalk"

    print("\nCreating raster + project...")
    raster = Raster(
        location=args.bbox,
        name="polygon_injection",
        zoom=19,
        base_tilesize=256
    )

    project = Project(
        "polygon_network_output",
        "outputs",
        raster
    )

    print("\nRunning topology engine...")
    pednet = PedNet(gdf, project)
    pednet.convert_whole_poly2line()

    network = pednet.complete_net

    end = time.time()

    print("\nSUCCESS")
    print("Edge count:", len(network))
    print("Runtime:", round(end - start, 2), "seconds")

    output_path = f"outputs/{args.output}.shp"
    network.to_file(output_path)

    print("Saved to:", output_path)
    print("\nDONE.")


if __name__ == "__main__":
    main()