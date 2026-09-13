# Causa site

Astro + Starlight docs site for Causa, deployed to Cloudflare Workers
(Static Assets).

## Develop

Requires [pnpm](https://pnpm.io) 12 (`packageManager` is pinned, corepack works too).

```bash
cd website
pnpm install
pnpm dev         # http://localhost:4321
```

## Build & preview the Worker locally

```bash
pnpm build
pnpm cf:dev      # serves dist/ through the Cloudflare worker
```

## Deploy

Pushes to `main` touching `website/**` deploy via
`.github/workflows/website.yml`. Required repo secret:

- `CLOUDFLARE_API_TOKEN` — a Cloudflare API token with
  **Workers Scripts: Edit** (and **Workers Routes** if a custom domain /
  route is attached). No account ID secret needed: it is pinned in
  `wrangler.jsonc`… unless you fork — then set `CLOUDFLARE_ACCOUNT_ID`
  or edit the file.

Manual deploy:

```bash
pnpm dlx wrangler login
pnpm deploy
```

## Custom domain

Attach it in the Cloudflare dashboard (Workers & Pages → this worker →
Settings → Domains & Routes), then set `site:` in `astro.config.mjs`
to the final URL.
