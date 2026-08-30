export interface Account {
  slug: string;
  riot_id: string;
  region: string;
}

export const ACCOUNTS: Account[] = [
  { slug: "spadzze", riot_id: "Spadzze#euw", region: "euw1" },
];

export function accountFor(slug: string): Account | undefined {
  return ACCOUNTS.find((account) => account.slug === slug);
}
