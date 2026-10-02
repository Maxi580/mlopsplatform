import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// `npm run dev` forwards API calls to a running API, e.g. MLP_API_URL=https://<domain>.
const api = process.env.MLP_API_URL ?? "http://localhost:8000";

export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(
      ["/auth", "/pipelines", "/datasets", "/schema", "/settings"].map((path) => [
        path,
        { target: api, changeOrigin: true, secure: false },
      ]),
    ),
  },
  test: { environment: "jsdom", globals: true, setupFiles: ["src/testSetup.ts"] },
});
