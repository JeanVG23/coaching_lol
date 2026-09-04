/** Helpers HTTP du Worker : réponses d'erreur, pagination, tranchage.
 *
 * `Response.json({ detail }, { status })` était écrit 21 fois (index/coach/feedback),
 * et la validation page/size existait en deux exemplaires avec des bornes différentes
 * (size <= 100 pour /reviews, <= 200 pour /games) — deux contrats pour le même
 * paramètre. Un seul endroit ici, la borne max reste paramétrable par route.
 */

export function jsonError(status: number, detail: string): Response {
  return Response.json({ detail }, { status });
}

export const notFound = (detail = "Not Found") => jsonError(404, detail);
export const methodNotAllowed = () => jsonError(405, "Method Not Allowed");
export const unprocessable = (detail: string) => jsonError(422, detail);

export interface Paging {
  page: number;
  size: number;
}

/** page/size validés, ou null si hors bornes (l'appelant répond 422). */
export function pageParams(params: URLSearchParams, maxSize = 100): Paging | null {
  const page = Number(params.get("page") ?? 1);
  const size = Number(params.get("size") ?? 20);
  const ok = Number.isInteger(page) && page >= 1
    && Number.isInteger(size) && size >= 1 && size <= maxSize;
  return ok ? { page, size } : null;
}

export function pagingError(maxSize = 100): Response {
  return unprocessable(`page>=1 et size in [1,${maxSize}]`);
}

export interface Page<T> {
  items: T[];
  page: number;
  size: number;
  total: number;
}

/** Tranche une liste déjà ordonnée selon page/size (1-indexé). */
export function paginate<T>(items: T[], { page, size }: Paging): Page<T> {
  const start = (page - 1) * size;
  return { items: items.slice(start, start + size), page, size, total: items.length };
}
