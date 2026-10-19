// Reads Rust code from tests/consumers/site-guides, the workspace CI
// compiles, so a page never carries a hand-copied program. `anchor` selects
// the lines between `// ANCHOR: name` and `// ANCHOR_END: name`; without
// it the whole file is returned. Marker lines are dropped and the common
// indent removed. A missing file or anchor fails the build.

const root = '../../../tests/consumers/site-guides/';
const sources = import.meta.glob('../../../tests/consumers/site-guides/src/**/*.rs', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const marker = /^\s*\/\/ ANCHOR(?:_END)?: \S+\s*$/;

/** `file` is a path inside tests/consumers/site-guides, e.g. `src/bin/home.rs`. */
export function snippet(file: string, anchor?: string): string {
  const source = sources[root + file];
  if (source === undefined) {
    throw new Error(`Snippet: no file ${file} in tests/consumers/site-guides`);
  }
  let lines = source.split('\n');
  if (anchor !== undefined) {
    const start = lines.findIndex((line) => line.trim() === `// ANCHOR: ${anchor}`);
    const end = lines.findIndex((line, i) => i > start && line.trim() === `// ANCHOR_END: ${anchor}`);
    if (start < 0 || end < 0) {
      throw new Error(`Snippet: no anchor ${anchor} in ${file}`);
    }
    lines = lines.slice(start + 1, end);
  }
  lines = lines.filter((line) => !marker.test(line));
  while (lines.length > 0 && lines[0].trim() === '') lines.shift();
  while (lines.length > 0 && lines[lines.length - 1].trim() === '') lines.pop();
  const indent = Math.min(
    ...lines.filter((line) => line.trim() !== '').map((line) => line.length - line.trimStart().length),
  );
  return lines.map((line) => line.slice(indent)).join('\n');
}
