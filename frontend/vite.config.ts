import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, Vite serves the UI on :5173 and proxies the backend (:8000).
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // reachable from a phone on the same Wi-Fi
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/s/": "http://localhost:8000",
    },
  },
});
