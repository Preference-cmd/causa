// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import cloudflare from '@astrojs/cloudflare';
import tailwindcss from '@tailwindcss/vite';
import starlightSidebarTopics from 'starlight-sidebar-topics';
import starlightLlmsTxt from 'starlight-llms-txt';

// Docs site: fully prerendered at build time, served by a tiny Worker
// (static assets + 404/redirect handling). No SSR, no bindings.
export default defineConfig({
  site: 'https://causa.inceptae.com',
  output: 'static',
  // Pages that moved when the site split into Guides and API Reference.
  // README files on crates.io and older links still point at these.
  redirects: {
    '/concepts': '/concepts/context-and-ports/',
    '/context-editing': '/guides/edit-context/',
    '/streaming': '/guides/stream-output/',
    '/interrupted-runs': '/guides/interrupted-runs/',
    '/crates': '/reference/crates/',
    '/status': '/reference/versioning/',
  },
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
      description:
        "Causa keeps an AI agent's conversation as typed Rust values your application owns. Your code runs the loop.",
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
      components: {
        // Adds the copy-as-Markdown links under every page title.
        PageTitle: './src/components/PageTitle.astro',
      },
      plugins: [
        // Two entries with their own sidebars: Guides say how to do
        // something, the API Reference says what each part is.
        starlightSidebarTopics(
          [
            {
              label: 'Guides',
              link: '/getting-started/',
              icon: 'open-book',
              items: [
                { label: 'Start', items: ['why-causa', 'getting-started'] },
                {
                  label: 'Concepts',
                  items: [
                    'concepts/context-and-ports',
                    'concepts/turns-and-batches',
                    'concepts/layering',
                  ],
                },
                {
                  label: 'Build',
                  items: [
                    'guides/write-a-tool',
                    'guides/connect-mcp-servers',
                    'guides/choose-a-provider',
                    'guides/save-and-restore',
                    'guides/edit-context',
                    'guides/prepare-each-request',
                    'guides/stream-output',
                    'guides/interrupted-runs',
                  ],
                },
                'examples',
              ],
            },
            {
              label: 'API Reference',
              link: '/reference/',
              icon: 'document',
              items: [
                {
                  label: 'Overview',
                  items: ['reference', 'reference/crates', 'reference/versioning'],
                },
                { label: 'causa-protocol', items: ['reference/protocol/stream-decoding'] },
                {
                  label: 'causa-provider',
                  items: ['reference/provider/protocol-options', 'reference/provider/stream-limits'],
                },
              ],
            },
          ],
          // The splash home page has no sidebar and belongs to neither.
          { exclude: ['/'] },
        ),
        starlightLlmsTxt({
          projectName: 'Causa',
          description:
            'Causa is a minimal context kernel for Rust agents: the kernel holds messages, tool calls and results as typed facts; the host application owns the loop, storage and policy.',
        }),
      ],
    }),
  ],
});
