/** The URL at the address this page was reached at, so a copied one works for whoever copies it. */
export function urlOnThisOrigin(url: string): string {
  return window.location.origin + pathOnThisOrigin(url);
}

/** The URL's path; the API names the platform's domain, which may not be how this page was reached. */
export function pathOnThisOrigin(url: string): string {
  const { pathname, search, hash } = new URL(url, window.location.href);
  return pathname + search + hash;
}
