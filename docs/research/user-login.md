# User login: how users log in, and how the API, KFP UI and MLflow UI are gated

Research ticket: [#20](https://github.com/Maxi580/mlopsplatform/issues/20) (map: [#1](https://github.com/Maxi580/mlopsplatform/issues/1)). Decisions belong to [#21](https://github.com/Maxi580/mlopsplatform/issues/21) ("What does user management look like in v1?"); this file gives options and a recommendation only.
Facts checked on **2026-10-01** against: Kubeflow Pipelines (KFP) **2.17.2**, MLflow **3.16.1**, oauth2-proxy **7.15.4**, Traefik **3.7.13** (k3s ships Traefik v3), Keycloak **26.8.0**, Dex **2.45.1**, Authelia **4.39.28**, authentik **2026.8.3**, ZITADEL **4.19.3**, Ory Kratos/Hydra **26.2.0**, cert-manager **1.21.2** [S1].

## Question

How do users log in, and how are the platform API, the Web UI, the KFP UI/API and the MLflow UI all gated behind that login, on one publicly reachable Azure VM running k3s? Sub-questions: identity option (built-in users vs self-hosted IdP vs Microsoft Entra ID), how the Web UI and CLI get tokens, how KFP and MLflow are fronted, whether the identity store fits in the shared Postgres, and what each option offers for account management without self-signup.

## Recommendation

**One login at the edge, one OIDC identity provider, one oauth2-proxy.**

1. **Every public hostname goes through Traefik with a `forwardAuth` middleware pointing at a single oauth2-proxy** (MIT [S1]): the Web UI, the platform API, the KFP UI and the MLflow UI. Traefik only forwards a request if oauth2-proxy answers 2xx [S10]; oauth2-proxy documents this exact Traefik setup [S8]. Sessions live in an encrypted cookie (the default `session_store_type=cookie` [S9]), so no Redis is needed. A cookie scoped to the parent domain gives single sign-on across all four hosts.
2. **The identity provider (IdP) issues identities; nothing in the platform stores passwords.** Two good options, picked by #21:
   - **Microsoft Entra ID** if the company already has an Entra tenant (likely, since it hosts on Azure). Zero run cost on the VM, MFA and offboarding come from the company directory, Entra ID Free is bundled with Azure subscriptions [S20]. Restrict the app with **"Assignment required = Yes"** and assign users one by one; *group* assignment needs Entra ID P1 [S19][S21].
   - **Keycloak** (Apache-2.0 [S1]) if the platform must own its users (no tenant access, or external users). Admin console for creating, disabling and resetting users, self-registration off by default, device grant, revocation endpoint [S13][S14], and it runs on the shared Postgres [S15]. Cost: roughly 1.25 GB RAM per pod [S16].
   Lighter self-hosted alternatives (Dex, Authelia) save RAM but have no admin UI for users (see Findings 1).
3. **CLI: OAuth 2.0 device authorization grant** (RFC 8628) against the IdP; Entra, Keycloak, Dex and Authelia all support it [S13][S18][S24][S26]. The CLI sends the access token as `Authorization: Bearer`. oauth2-proxy lets verified JWTs through with `--skip-jwt-bearer-tokens` (plus `--extra-jwt-issuers` for the CLI's client ID) [S9], and the API validates the JWT again itself, reading the Owner from the `sub`/`oid` claim. If the tenant blocks device code flow via Conditional Access (Microsoft recommends blocking it where possible [S22]), use authorization code + PKCE with a `localhost` redirect instead.
4. **KFP API: do not expose `ml-pipeline` itself.** The platform API talks to it in-cluster. But note the KFP UI server **proxies the whole KFP API** (`/apis/v1beta1/*`, `/apis/v2beta1/*`) to `ml-pipeline` [S4], so the KFP UI host is effectively the KFP API and must be behind the login. Optionally make it read-only at the edge (see Findings 3).
5. **MLflow: no second login.** Run the MLflow server **without** `--app-name basic-auth`, gated only by the edge login, and set the KFP MLflow plugin to `authType: none` [S3]. A NetworkPolicy should then limit in-cluster access to MLflow to the KFP namespace, the platform API and Stage pods. Use MLflow's `basic-auth` (users in Postgres) or the community `mlflow-oidc-auth` plugin (Apache-2.0) only if #21 wants per-user permissions inside MLflow [S5][S6].
6. **Postgres:** the self-hosted IdPs all fit the one Postgres as their own database (`keycloak` or `dex`, next to `platform`, `mlflow`) [S15][S25][S31]. Entra ID stores nothing locally; oauth2-proxy stores nothing.

Cheapest secure setup on one VM: **Entra ID + oauth2-proxy** (one small pod, ~no state). Most self-contained: **Keycloak + oauth2-proxy** (one ~1.5 GB pod plus a Postgres database).

## Findings

### 1. Identity options

| Option | Licence (flags) | Runs on the VM | Postgres | User management without self-signup | CLI device flow | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **Built-in users in the platform API** (Postgres + Argon2 hashes, API issues tokens) | our code; libraries are MIT/BSD | inside the API | yes, `platform` DB | whatever we build (create, disable, reset, PATs) | we would build it, or use PATs | We own password storage, MFA, brute-force protection, reset flows. KFP and MLflow UIs still need gating: the API would have to implement the forwardAuth endpoint itself. Most code, weakest security for least benefit. |
| **Microsoft Entra ID** | proprietary SaaS; Free tier bundled with Azure [S20] | nothing (SaaS) | n/a | company directory; restrict app via "Assignment required" + per-user assignment (free) or group assignment (P1, $7/user/month) [S19][S20][S21] | yes [S18] | Needs a tenant admin to register the app and grant consent [S19]. Conditional Access may block device code flow [S22]. Users disabled in the directory lose access at next token refresh. Guests need B2B invites. |
| **Keycloak** | Apache-2.0 [S1]; images on quay.io | one JVM pod, ~1250 MB base memory [S16] | yes, PostgreSQL 14 to 18 [S15] | full admin console: create users, set temporary passwords, required actions (reset password, configure OTP), disable; registration is a per-realm toggle [S14] | yes [S13] | Token revocation endpoint, session management, not-before revocation per realm/client/user [S13][S14]. Can also broker to Entra later. |
| **Dex** | Apache-2.0 [S1] | small Go binary | yes (also SQLite, CRDs) [S25] | local users via `staticPasswords` in config or the gRPC API; no admin UI, no reset flow [S23] | yes (`device_code` grant) [S24] | Mainly a federation broker (designed to sit in front of other IdPs). Last release March 2026. |
| **Authelia** | Apache-2.0 [S1] | small Go binary | yes, for its state; users live in a YAML/TOML/JSON file or LDAP [S27][S28] | edit the users file (hot-reload with `watch`); no admin UI or API [S27] | yes [S26] | Native Traefik forwardAuth (`/api/authz/forward-auth`) [S29], OpenID certified, but its OIDC provider is still labelled "open beta" [S26]. |
| **authentik** | MIT core, **except `authentik/enterprise/`** under a separate enterprise licence [S2] | needs 2 CPU / 2 GB RAM [S30] | yes; since 2025.10 PostgreSQL only, no Redis [S31] | admin UI, flows, invitations | not re-checked here | Fine if enterprise features are not used; heavier than Keycloak's single pod on paper. |
| **ZITADEL** | **AGPL-3.0-only** since v3 [S1][S32] | Go binary | yes | admin console | not re-checked here | **Flagged: AGPL.** Vendor recommends legal review and sells commercial licences [S32]. Excluded by the map's licensing rule unless approved. |
| **Ory Kratos + Hydra** | Apache-2.0 [S1] | two Go services | yes | Kratos admin API; no ready-made UI | not re-checked here | You must build or host the login/consent UI yourself. Highest operating effort. |
| **Kanidm**, **Pocket ID** | MPL-2.0, BSD-2-Clause [S1] | small | Kanidm: own DB; Pocket ID: SQLite/Postgres | basic | not checked | Not evaluated in depth. Pocket ID is passkey-only. |

### 2. Tokens and sessions

- **Web UI:** no token handling in the browser. oauth2-proxy runs the OIDC authorization code flow and stores the session in an encrypted cookie; default lifetime `cookie_expire=168h`, and `cookie_refresh` re-validates with the IdP (supported by Entra, Keycloak and any full OIDC provider) [S9]. A short `cookie_refresh` (e.g. 1h) is what makes "disable the user at the IdP" take effect on browser sessions. oauth2-proxy can pass identity to the upstream as `X-Forwarded-User`/`X-Forwarded-Email`, as the access token (`--pass-access-token`, `X-Forwarded-Access-Token`) or as the ID token in `Authorization` (`--pass-authorization-header`) [S9]. With Traefik, these come back as `authResponseHeaders` [S10], which must be listed explicitly; Traefik overwrites any client-supplied header of the same name [S10], so the API can trust them only when it is reachable solely through Traefik.
- **CLI:** device authorization grant: the CLI shows a code and URL, the user signs in in a browser, the CLI polls the token endpoint and gets access + refresh tokens [S18]. Entra access tokens last about an hour (`expires_in: 3599` in Microsoft's example) [S18]. The refresh token is stored in the user's OS keyring or a `0600` file.
- **Personal access tokens (PATs):** none of the IdPs above issue PATs to end users. For CI or scripts, options are (a) refresh tokens / Keycloak offline tokens, (b) IdP client credentials (Keycloak, Entra, Dex behind a feature flag [S24]), or (c) PATs minted by the platform API and stored hashed in the `platform` DB. Option (c) means the API host must accept a non-JWT bearer, so oauth2-proxy can't be the only check on that host (use `--api-route` or let the API host skip oauth2-proxy and validate everything itself).
- **Revocation:** access tokens are stateless JWTs and stay valid until expiry; keep them short (≤1h). Revoke by disabling the user at the IdP (refresh fails), by Keycloak's revocation endpoint and not-before policies [S13][S14], or for platform PATs by deleting the row.

### 3. Fronting KFP and MLflow

- **KFP standalone has no authentication.** Its UI server has an `auth.enabled` mode that only checks a `kubeflow-userid` header in multi-user deployments [S4]; standalone single-user (chosen in #2) has none. The UI server proxies `/apis/v1beta1/*` and `/apis/v2beta1/*` to the `ml-pipeline` API server, plus artifacts, pod logs and the metadata Envoy [S4]. Exposing the KFP UI therefore exposes the full KFP API, including creating runs with arbitrary containers (and GPUs).
- **So: gate the KFP UI host with forwardAuth, and don't add an Ingress for `ml-pipeline` itself.** The platform API reaches `ml-pipeline` in-cluster. Optional hardening for #21: a Traefik router that allows only `GET` on `/apis/*` makes the KFP UI read-only (graph and logs still work), at the cost of the UI's terminate/retry buttons; cancelling then goes through the platform API.
- **MLflow** has built-in `basic-auth` (`mlflow server --app-name basic-auth`): users and RBAC permissions in a SQL DB, which can be Postgres (`database_uri`; auth migrations use their own `alembic_version_auth` table, so it can even share the `mlflow` DB), a mandatory admin password of 12+ characters, a REST/Python API to manage users [S5]. It is a separate username/password login, not SSO. For SSO, MLflow points to a reverse proxy or the community `mlflow-oidc-auth` plugin (Apache-2.0, "maintained by the community") [S6]. MLflow 3.5+ also has `--allowed-hosts` and `--cors-allowed-origins` middleware, which must be set to the public hostname when it sits behind an ingress [S7].
- **KFP's MLflow plugin** authenticates to MLflow with `authType` `kubernetes` (needs the Kubeflow/MLflow integration with kube-rbac-proxy-style auth [S3][S11]), `basic-auth` or `bearer` (credentials from a fixed Secret `kfp-mlflow-credentials`), or `none` [S3]. With edge-only gating, in-cluster traffic skips Traefik, so `none` works; if MLflow `basic-auth` is turned on, use `basic-auth` with a service user.
- **Header-based identity for MLflow:** MLflow's `authorization_function` is pluggable, so a small function could trust oauth2-proxy's `X-Forwarded-Email` header [S5]. MLflow labels its JWT example "not intended for production use" [S5]; treat this as custom code.

### 4. Postgres

- Keycloak: PostgreSQL 14 to 18 supported [S15]. Dex: Postgres, MySQL, SQLite, etcd or CRDs [S25]. Authelia: Postgres for its state, users stay in a file or LDAP [S27][S28]. authentik: Postgres only since 2025.10 (about 50% more connections than before) [S31]. MLflow basic-auth: any SQLAlchemy DB [S5].
- All fit the "one Postgres server, one database per tool" layout from #3. oauth2-proxy with cookie sessions and Entra ID need no database.

### 5. TLS and ingress

- k3s deploys Traefik v3, customised through a `HelmChartConfig` in `/var/lib/rancher/k3s/server/manifests` [S12]. ForwardAuth is a Traefik `Middleware` CRD (`traefik.io/v1alpha1`) attached to each IngressRoute/Ingress [S10]. oauth2-proxy must run with `--reverse-proxy` [S8]. Use the `errors` middleware with `statusRewrites 401→302` or `--upstream=static://202` so browsers are redirected to the sign-in page [S8].
- cert-manager (Apache-2.0 [S1]) + Let's Encrypt covers TLS for every host (decided in #9). Wildcard certs need a DNS-01 solver; per-host certs with HTTP-01 are enough.

## Licence summary

| Component | Licence | Flag |
| --- | --- | --- |
| oauth2-proxy | MIT [S1] | none |
| Traefik (k3s bundled) | MIT [S1] | none |
| cert-manager | Apache-2.0 [S1] | none |
| Keycloak | Apache-2.0 [S1] | none |
| Dex | Apache-2.0 [S1] | none |
| Authelia | Apache-2.0 [S1] | OIDC provider in open beta [S26] |
| authentik | MIT + separate enterprise licence for `authentik/enterprise/` [S2] | don't use enterprise features |
| ZITADEL | AGPL-3.0-only [S32] | **AGPL** |
| Ory Kratos/Hydra | Apache-2.0 [S1] | none |
| mlflow-oidc-auth | Apache-2.0 [S1] | community-maintained [S6] |
| Microsoft Entra ID | proprietary SaaS, Free tier [S20] | P1 needed for group assignment and Conditional Access [S20][S21] |

## Open for #21

- Gate only (every logged-in user sees everything) vs per-user visibility of Pipelines, KFP runs and MLflow Runs. KFP standalone can't do per-user visibility at all; MLflow can (basic-auth or OIDC plugin).
- Admin role(s), and who creates, disables and resets users.
- Entra ID vs self-hosted Keycloak (depends on tenant access and whether non-employees need accounts).
- Read-only KFP UI or not; whether PATs are needed for CI/scripts.

## Sources

- [S1] GitHub repository metadata (licence, latest release), queried 2026-10-01: [oauth2-proxy](https://github.com/oauth2-proxy/oauth2-proxy), [traefik](https://github.com/traefik/traefik), [cert-manager](https://github.com/cert-manager/cert-manager), [keycloak](https://github.com/keycloak/keycloak), [dex](https://github.com/dexidp/dex), [authelia](https://github.com/authelia/authelia), [authentik](https://github.com/goauthentik/authentik), [zitadel](https://github.com/zitadel/zitadel), [ory/kratos](https://github.com/ory/kratos), [ory/hydra](https://github.com/ory/hydra), [kanidm](https://github.com/kanidm/kanidm), [pocket-id](https://github.com/pocket-id/pocket-id), [mlflow-oidc-auth](https://github.com/mlflow-oidc/mlflow-oidc-auth), [mlflow](https://github.com/mlflow/mlflow/releases/tag/v3.16.1)
- [S2] authentik LICENSE: https://github.com/goauthentik/authentik/blob/main/LICENSE
- [S3] KFP MLflow plugin operator guide: https://github.com/kubeflow/website/blob/master/content/en/docs/components/pipelines/operator-guides/mlflow-plugin.md
- [S4] KFP UI server (2.17.2), API/metadata/artifact proxies and auth middleware: https://github.com/kubeflow/pipelines/blob/2.17.2/frontend/server/app.ts
- [S5] MLflow basic HTTP auth (v3.16.1): https://github.com/mlflow/mlflow/blob/v3.16.1/docs/docs/self-hosting/security/basic-http-auth.mdx
- [S6] MLflow SSO (v3.16.1): https://github.com/mlflow/mlflow/blob/v3.16.1/docs/docs/self-hosting/security/sso.mdx
- [S7] MLflow network security middleware: https://github.com/mlflow/mlflow/blob/v3.16.1/docs/docs/self-hosting/security/network.mdx
- [S8] oauth2-proxy Traefik integration: https://github.com/oauth2-proxy/oauth2-proxy/blob/master/docs/docs/configuration/integrations/traefik.md
- [S9] oauth2-proxy configuration overview (cookie, session store, JWT bearer, headers): https://github.com/oauth2-proxy/oauth2-proxy/blob/master/docs/docs/configuration/overview.md
- [S10] Traefik ForwardAuth middleware: https://doc.traefik.io/traefik/reference/routing-configuration/http/middlewares/forwardauth/
- [S11] MLflow Kubernetes authentication: https://github.com/mlflow/mlflow/blob/v3.16.1/docs/docs/self-hosting/security/kubernetes.mdx
- [S12] k3s networking services (Traefik): https://docs.k3s.io/networking/networking-services
- [S13] Keycloak OIDC layers (grants, device grant, revocation): https://www.keycloak.org/securing-apps/oidc-layers
- [S14] Keycloak Server Administration Guide: https://www.keycloak.org/docs/latest/server_admin/index.html
- [S15] Keycloak database configuration: https://www.keycloak.org/server/db
- [S16] Keycloak memory and CPU sizing: https://www.keycloak.org/high-availability/multi-cluster/concepts-memory-and-cpu-sizing
- [S18] Microsoft identity platform device authorization grant: https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-device-code
- [S19] Restrict a Microsoft Entra app to a set of users: https://learn.microsoft.com/en-us/entra/identity-platform/howto-restrict-your-app-to-a-set-of-users
- [S20] Microsoft Entra pricing: https://www.microsoft.com/en-us/security/business/microsoft-entra-pricing
- [S21] Assign users and groups to an application (group assignment needs P1/P2): https://learn.microsoft.com/en-us/entra/identity/enterprise-apps/assign-user-or-group-access-portal
- [S22] Block authentication flows with Conditional Access: https://learn.microsoft.com/en-us/entra/identity/conditional-access/policy-block-authentication-flows
- [S23] Dex local connector: https://dexidp.io/docs/connectors/local/
- [S24] Dex OAuth2 configuration (grant types incl. device code): https://github.com/dexidp/website/blob/main/content/docs/configuration/oauth2.md
- [S25] Dex storage: https://dexidp.io/docs/configuration/storage/
- [S26] Authelia OpenID Connect provider: https://www.authelia.com/integration/openid-connect/introduction/
- [S27] Authelia file authentication backend: https://www.authelia.com/configuration/first-factor/file/
- [S28] Authelia PostgreSQL storage: https://www.authelia.com/configuration/storage/postgres/
- [S29] Authelia Traefik integration: https://www.authelia.com/integration/proxies/traefik/
- [S30] authentik Docker Compose requirements: https://docs.goauthentik.io/install-config/install/docker-compose/
- [S31] authentik 2025.10 release notes (Redis removed): https://docs.goauthentik.io/releases/2025.10/
- [S32] ZITADEL LICENSING.md: https://github.com/zitadel/zitadel/blob/main/LICENSING.md
