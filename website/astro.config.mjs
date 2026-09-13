// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import cloudflare from '@astrojs/cloudflare';
import tailwindcss from '@tailwindcss/vite';

// Docs site: fully prerendered at build time, served by a tiny Worker
// (static assets + 404/redirect handling). No SSR, no bindings.
export default defineConfig({
  // TODO: set to the final custom domain, e.g. https://causa.dev
  site: 'https://causa-site.workers.dev',
  output: 'static',
  vite: {
    plugins: [tailwindcss()],
  },
  adapter: cloudflare({
    // Fully static docs: prerender in Node instead of workerd.
    // Identical HTML, avoids bundling workerd-only WASM shims at build time.
    prerenderEnvironment: 'node',
  }),
  integrations: [
    starlight({
      title: 'Causa',
      description: 'A small, principled agent kernel for Rust — facts, ports, driver.',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/Preference-cmd/causa',
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/Preference-cmd/causa/edit/main/website/',
      },
      customCss: ['./src/styles/tailwind.css', './src/styles/custom.css'],
      sidebar: [
        { label: 'Getting started', slug: 'getting-started' },
        { label: 'Concepts', slug: 'concepts' },
        { label: 'Crates', slug: 'crates' },
        { label: 'Examples', slug: 'examples' },
        { label: 'Status & changelog', slug: 'status' },
      ],
    }),
  ],
});
