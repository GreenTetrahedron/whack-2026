import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// envDir ".." so VITE_API_URL is read from the repo-root .env (Vite only exposes VITE_* vars).
export default defineConfig({
  plugins: [react()],
  envDir: "..",
  server: { port: 5173 },
});
