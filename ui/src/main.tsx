// ui/src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { loadRuntimeConfig, UiRuntimeConfig } from "./lib/runtime_config";
import { setApiBaseUrl } from "./lib/api";

async function bootstrap() {
  const runtimeConfig: UiRuntimeConfig = await loadRuntimeConfig();

  // Wire runtime config into API client (supports /config.json at runtime)
  setApiBaseUrl(runtimeConfig.api?.base_url);

  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <App runtimeConfig={runtimeConfig} />
    </React.StrictMode>
  );
}

bootstrap();