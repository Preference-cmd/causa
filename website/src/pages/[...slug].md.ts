// `/<page>.md`: every docs page as Markdown, built with the site. The home
// page is `/index.md`. PageTitle's copy button fetches these files.
import type { APIRoute, GetStaticPaths } from 'astro';
import { getCollection, type CollectionEntry } from 'astro:content';
import { pageMarkdown } from '../lib/page-markdown';

export const prerender = true;

export const getStaticPaths = (async () => {
  const entries = await getCollection('docs', (entry) => !entry.data.draft && entry.id !== '404');
  return entries.map((entry) => ({ params: { slug: entry.id }, props: { entry } }));
}) satisfies GetStaticPaths;

export const GET: APIRoute<{ entry: CollectionEntry<'docs'> }> = async (context) =>
  new Response(await pageMarkdown(context.props.entry, context), {
    headers: { 'Content-Type': 'text/markdown; charset=utf-8' },
  });
