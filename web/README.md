# Counterfactual XRL — interactive viewer

Static web app for stepping through the
[counterfactual-xrl](../README.md) results side-by-side (PPO vs MCTS),
hosted at <https://counterfactualxrl.mynewapi.com>.

## Layout

```
web/
├── README.md          ← this file
├── build_bundle.py    ← packs records + explanations → site/data.json
└── site/              ← static SPA served by nginx
    ├── index.html
    ├── styles.css
    ├── app.js
    └── data.json      ← bundle (rebuild after a fresh judge run)
```

## Rebuilding after new results

```bash
.venv/bin/python web/build_bundle.py        # rewrite site/data.json
```

## Running locally

```bash
cd web/site && python -m http.server 8765
# http://localhost:8765
```

## Deploying to counterfactualxrl.mynewapi.com (AgentOps platform)

The deployed copy lives at `/srv/agentops/stacks/counterfactualxrl/`
on the host. Push updates with:

```bash
sudo cp -r web/site /srv/agentops/stacks/counterfactualxrl/site
sudo /srv/agentops/bin/agentops-deploy counterfactualxrl --restart
```

The first-time setup added:

* `/srv/agentops/stacks/counterfactualxrl/{compose.yaml,nginx.conf,site/}`
  — the nginx-alpine container and config
* a new entry in `/srv/agentops/secrets/cloudflared/config.yml`
  routing `counterfactualxrl.mynewapi.com → http://localhost:80`
* `cloudflared tunnel route dns agentops counterfactualxrl.mynewapi.com`
* a file-based Traefik route at
  `/srv/agentops/volumes/traefik/dynamic/counterfactualxrl.yaml`
  (the docker provider on this host did not pick up the in-compose
  `traefik.*` labels for this stack — root cause unknown; the file
  provider works around it cleanly)

## Features

* Synced trajectory + step navigation across both agents
* Side-by-side 8×8 grid renderer with agent direction + obstacles + goal
* Per-step rationale, counterfactual, and confidence
* Inline highlighting of every numeric claim against the evidence:
  green if within ε = 0.1, red if hallucinated
* Per-action evidence table (discounted return, CI, std, steps, n)
* MCTS-only "tree diagnostics" collapsible panel (principal variations,
  depth-2 visit distribution, per-child variance)
* Keyboard shortcuts: ←/→ for prev/next, space to play/pause
