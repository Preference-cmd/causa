// Turns a docs page into Markdown for the "Copy as Markdown" button and the
// `/<page>.md` routes. The page is rendered to HTML first and converted
// back, so components such as <Snippet> contribute the code they render
// instead of their raw MDX tags.
import mdxServer from '@astrojs/mdx/server.js';
import type { APIContext } from 'astro';
import { experimental_AstroContainer } from 'astro/container';
import { render, type CollectionEntry } from 'astro:content';
import rehypeParse from 'rehype-parse';
import rehypeRemark from 'rehype-remark';
import remarkGfm from 'remark-gfm';
import remarkStringify from 'remark-stringify';
import { unified } from 'unified';

interface HastNode {
  type: string;
  tagName?: string;
  value?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
}

function classes(node: HastNode): string[] {
  const value = node.properties?.className;
  return Array.isArray(value) ? value.map(String) : [];
}

function text(node: HastNode): string {
  if (node.type === 'text') return node.value ?? '';
  return (node.children ?? []).map(text).join('');
}

// Page chrome that reads as noise in Markdown: heading anchor links, the
// code block copy button, screen-reader labels in code block headers and
// HTML comments.
function isChrome(node: HastNode, parent: HastNode): boolean {
  if (node.type === 'comment') return true;
  if (node.type !== 'element') return false;
  if (node.tagName === 'button') return true;
  if (classes(node).includes('sl-anchor-link')) return true;
  return parent.tagName === 'figcaption' && classes(node).includes('sr-only');
}

function tidy(node: HastNode, site: URL): void {
  if (!node.children) return;
  node.children = node.children.filter((child) => !isChrome(child, node));
  for (const child of node.children) tidy(child, site);
  // A site-relative link would break once the Markdown is pasted elsewhere.
  const href = node.tagName === 'a' ? node.properties?.href : undefined;
  if (typeof href === 'string' && href.startsWith('/')) {
    node.properties = { ...node.properties, href: new URL(href, site).href };
  }
  // Expressive Code wraps each line in its own block element, which the
  // converter would separate with extra blank lines, and puts the language
  // on <pre>. Rebuild <code> as plain lines with a `language-*` class.
  const language = node.tagName === 'pre' ? node.properties?.dataLanguage : undefined;
  const code = node.children.find((child) => child.tagName === 'code');
  if (typeof language === 'string' && code) {
    const lines = (code.children ?? [])
      .filter((child) => classes(child).includes('ec-line'))
      .map((line) => text(line).replace(/\n$/, ''));
    code.properties = { ...code.properties, className: [`language-${language}`] };
    code.children = [{ type: 'text', value: lines.join('\n') }];
  }
}

function converter(site: URL) {
  return unified()
    .use(rehypeParse, { fragment: true })
    .use(() => (tree: HastNode) => tidy(tree, site))
    .use(rehypeRemark)
    .use(remarkGfm)
    .use(remarkStringify);
}

const container = await experimental_AstroContainer.create({
  renderers: [{ name: 'astro:jsx', ssr: mdxServer }],
});

/** The page's URL path, for example `/guides/edit-context/` or `/`. */
export function pagePath(entry: CollectionEntry<'docs'>): string {
  return entry.id === 'index' ? '/' : `/${entry.id}/`;
}

/** Renders one docs page as a Markdown document headed by its title. */
export async function pageMarkdown(entry: CollectionEntry<'docs'>, context: APIContext): Promise<string> {
  const { Content } = await render(entry);
  const html = await container.renderToString(Content, context);
  const site = new URL(context.site ?? 'https://causa.inceptae.com');
  const body = String(await converter(site).process(html)).trim();
  const title = entry.data.hero?.title || entry.data.title;
  const description = entry.data.hero?.tagline || entry.data.description;
  const url = new URL(pagePath(entry), site).href;
  const head = [`# ${title}`];
  if (description) head.push(`> ${description}`);
  head.push(`URL: ${url}`);
  return `${head.join('\n\n')}\n\n${body}\n`;
}
