import type { Endpoint } from "../serving/endpoint";
import type { Dataset, RegisteredModel } from "../storage/storage";

// What a page's pickers offer, loaded once for all of them.
export type Catalog = {
  models: RegisteredModel[];
  endpoints: Endpoint[];
  datasets: Dataset[];
  // The user's Hugging Face token, so a search also finds the models it opens.
  hfToken?: string;
};

export const NO_CATALOG: Catalog = { models: [], endpoints: [], datasets: [] };
