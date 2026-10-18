import { defineCollection } from 'astro:content';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';

export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
  // Starlight's UI strings. The site keeps the English defaults; the empty
  // en.json only gives the collection an entry, so the build does not warn
  // that it is missing.
  i18n: defineCollection({ loader: i18nLoader(), schema: i18nSchema() }),
};
