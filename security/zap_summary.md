# OWASP ZAP Security Scan

## How to run a baseline scan

### Prerequisites

- Docker installed and running
- The intake-api running locally on port 8001 (or deployed to AKS)

### Pull the ZAP image

```bash
docker pull ghcr.io/zaproxy/zaproxy:stable
```

### Run against local service (PowerShell)

```powershell
docker run --rm -v "${PWD}:/zap/wrk/:rw" ghcr.io/zaproxy/zaproxy:stable `
  zap-baseline.py `
  -t http://host.docker.internal:8001 `
  -r zap_report.html
```

### Run against local service (Mac/Linux)

```bash
docker run --rm -v "$(pwd):/zap/wrk/:rw" ghcr.io/zaproxy/zaproxy:stable \
  zap-baseline.py \
  -t http://host.docker.internal:8001 \
  -r zap_report.html
```

### Run against AKS ingress

```bash
docker run --rm -v "$(pwd):/zap/wrk/:rw" ghcr.io/zaproxy/zaproxy:stable \
  zap-baseline.py \
  -t http://<EXTERNAL-IP> \
  -r zap_report.html
```

The report is saved to `zap_report.html` in the current directory.

---

## Scan summary template

Fill this in after running the scan.

| Field | Value |
|---|---|
| **Scan target** | `http://localhost:8001` |
| **Scan date** | YYYY-MM-DD |
| **ZAP version** | stable |
| **Scan type** | Baseline |

### Alerts found and status

| Alert | Risk | Status |
|---|---|---|
| Missing Anti-clickjacking Header | Medium | Fixed — `X-Frame-Options: DENY` added |
| X-Content-Type-Options Header Missing | Low | Fixed — `nosniff` header added |
| Content Security Policy (CSP) Header Not Set | Medium | Accepted risk — CSP deferred to Ingress/WAF layer |
| Absence of Anti-CSRF Tokens | Medium | Accepted risk — stateless JWT API; no session cookies |
| Server Leaks Version Information | Low | Under review |

### What was implemented to address ZAP findings

1. **Security headers** — `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy` added via FastAPI middleware on every response.
2. **Rate limiting** — `/v1/login` capped at 5 req/min per IP; intake routes at 20–30 req/min.
3. **JWT authentication** — all data routes require a valid Bearer token.
4. **RBAC** — roles enforced per-route (`admin`, `user`, `service`).
5. **CORS** — restricted to `ALLOWED_ORIGINS` env var (not wildcard `*`).
6. **Input validation** — Pydantic enforces field types and ranges on all request bodies.
7. **Payload size limit** — requests over 1 MB rejected with 413 before parsing.

### Accepted risks

- **CSP header**: Not set at the application layer. Should be configured on the Azure Application Gateway or Ingress controller in production.
- **HTTPS**: Not enforced at the application layer. TLS termination handled by AKS Ingress + cert-manager or Azure Application Gateway.
