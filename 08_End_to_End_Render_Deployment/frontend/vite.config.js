import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    // While developing with `npm run dev`, forward API calls to Flask
    // running on port 5000 (start Flask separately with `python app.py`).
    proxy: {
      "/api": "http://127.0.0.1:5000",
    },
  },
});
