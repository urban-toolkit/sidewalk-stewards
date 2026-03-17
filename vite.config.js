import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

function saveNetworkPlugin() {
  return {
    name: "save-network",
    configureServer(server) {
      server.middlewares.use("/api/save-network", (req, res, next) => {
        if (req.method !== "POST") return next();

        let body = "";
        req.on("data", (chunk) => (body += chunk));
        req.on("end", () => {
          try {
            const geojson = JSON.parse(body);
            if (geojson.type !== "FeatureCollection" || !Array.isArray(geojson.features)) {
              res.statusCode = 400;
              res.end("Invalid GeoJSON: expected a FeatureCollection");
              return;
            }

            const outPath = path.resolve("public", "network.geojson");
            fs.writeFileSync(outPath, JSON.stringify(geojson), "utf-8");

            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true, features: geojson.features.length }));
          } catch (err) {
            console.error("Save network error:", err);
            res.statusCode = 500;
            res.end(String(err));
          }
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), saveNetworkPlugin()],
  server: {
    proxy: {
      "/tiles": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/api/train": {
        target: "http://localhost:8001",
        changeOrigin: true,
      },
    },
  },
});