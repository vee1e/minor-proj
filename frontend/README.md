# satsel frontend

Research dashboard for the satsel solver-selection pipeline (React 19 + TypeScript + Vite + Tailwind 3 + recharts).

    npm install
    npm run dev      # http://localhost:5173, proxies /api -> http://127.0.0.1:5001 (web_server.py)
    npm run build    # tsc -b && vite build -> dist/, served by web_server.py

Structure:

- `src/lib/api.ts` — typed fetch client; 404 is surfaced as `ApiError.isNotFound` so views show "run step X first".
- `src/lib/types.ts` — response shapes for every endpoint.
- `src/lib/palette.ts` — one categorical palette (CSS tokens `--chart-1..7`) and stable per-solver colours.
- `src/hooks/` — `useApi` (load/reload/poll), `usePipeline` (status polling + run-step), `useTheme`.
- `src/components/views/` — Overview (pipeline stepper + log drawer), Data, Results, Diagnostics, Oracle.
- `src/components/charts/Charts.tsx` — shared recharts wrappers.
- `src/components/Primitives.tsx` — page header, section card, callouts, loading/empty/error states, stat tiles.
