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
  route is attached). The checked-in `wrangler.jsonc` does not pin an
  account ID, and the workflow passes only the token to Wrangler. Ensure
  that token identifies the intended deployment account. If your setup
  needs an explicit account ID, configure it before deploying; there is no
  repository-provided account selection to inherit when forking.

Manual deploy:

```bash
pnpm dlx wrangler login
pnpm deploy
```

## Custom domain

The site is served at https://causa.inceptae.com, attached in the
Cloudflare dashboard (Workers & Pages → this worker → Settings → Domains &
Routes). `site:` in `astro.config.mjs` must match it.

## Writing pages

The site has two topics, each with its own sidebar (`astro.config.mjs`):
**Guides** say how to do something, the **API Reference** says what each
part is. Pages under `src/content/docs/concepts/` and `guides/` belong to
Guides; pages under `reference/` belong to the API Reference. A new page
needs an entry in the topic's sidebar, and a moved page needs a redirect.

### Guide template

Sections in this order; leave out a section with nothing to say rather than
keeping it empty.

1. A `<RunIt>` box under the title (see below).
2. **What it does** — one paragraph.
3. **When to reach for it** — and when not to.
4. **Before you start** — the features, keys and earlier guides it needs.
5. **The decisions** — the choices the host makes, as a table of options
   and defaults.
6. **Code** — `<Snippet>` blocks only.
7. **Common questions** — phrased the way someone evaluating Causa asks
   them.
8. **It's working if** — a list of things the reader can observe.
9. **Where it fits** — neighbouring guides, the example, docs.rs items.

### Reference template

Each page names, near the top, the source files its facts come from.

1. **Purpose** — one paragraph.
2. **Key items** — a table: name, one line, docs.rs link.
3. **Guarantees** — invariants, defaults, option tables.
4. **Failure modes** — errors and interruptions.
5. **Stability** — whether the type is in the stable serialized set and
   whether it is `#[non_exhaustive]`, i.e. whether a new variant or field
   breaks callers.
6. **Used in** — the guides that use it.

Signatures are not copied onto the site; link the docs.rs page instead.

### Components

- `Snippet` — shows Rust code from `tests/consumers/site-guides`, which CI
  compiles. Never write a Rust block in a page; add the code there and
  mark the shown lines with `// ANCHOR: name` / `// ANCHOR_END: name`.

  ```mdx
  <Snippet file="src/bin/context_editing.rs" anchor="program" />
  ```

- `RunIt` — the box at the top of a guide: the example to run, or the
  `cargo add` line when there is no example, then links to the example
  source and the docs.rs pages, given as paths after `https://docs.rs/`.

  ```mdx
  <RunIt
    command="cargo run --example streaming_print -p causa-runtime"
    example="crates/causa-runtime/examples/streaming_print.rs"
    docs={['causa-runtime/latest/causa_runtime/driver/struct.TurnRunner.html']}
  />
  ```

### Markdown for agents

The build also writes `/llms.txt`, `/llms-full.txt` and `/llms-small.txt`
(`starlight-llms-txt`), and every page as `/<page>.md`
(`src/pages/[...slug].md.ts`). The "Copy as Markdown" button under each
title copies that file. Both render the page to HTML and convert it back,
so components contribute what they show; a new component should produce
HTML that reads sensibly as Markdown.
