import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles/index.css";

// No <StrictMode>: the session holds one imperative WebSocket, and StrictMode's
// dev-only double-mount would open it twice. The shipped production build never
// double-mounts; correctness is covered by the store's unit tests instead.
const root = document.getElementById("root");
if (root) createRoot(root).render(<App />);
