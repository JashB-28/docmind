# Deploying DocMind (EC2 + Docker Hub, HTTPS via Caddy)

DocMind ships as one image (`jash09/docmind:latest`) that serves the API and the
built SPA. You pull it on the box and front it with Caddy for HTTPS. Two layouts:

- **Behind an existing reverse proxy** (another app already owns 80/443) — join
  that proxy's network and add a subdomain. *(This is the common case.)*
- **Standalone** (a fresh box) — DocMind runs its own Caddy on 80/443.

Bedrock works on the box via an **EC2 instance role** — no AWS keys on disk.

---

## Prerequisites on the box (one-time)
- Docker + the Compose plugin.
- This repo cloned to `/opt/docmind` (or anywhere; adjust paths).
- A `.env` at the repo root — copy `.env.example`, set `PINECONE_API_KEY`,
  `DEFAULT_PROVIDER=bedrock`, `ENABLE_OLLAMA=false`, and `AWS_REGION`.
- An **IAM instance role** attached with `AmazonBedrockFullAccess`
  (+ `AmazonSSMManagedInstanceCore` for the CD pipeline).
- **Instance metadata hop limit = 2** (EC2 → Instance → Actions → Instance
  settings → Modify instance metadata options) so the container can read the
  role. Without this, Bedrock fails from inside Docker.

## Optional — S3 source-document storage
To store uploaded PDFs in S3 (presigned downloads, auto-expiry):
1. Create a private bucket (e.g. `docmind-uploads-<unique>`) in `AWS_REGION`, and
   add a **lifecycle rule** expiring objects after ~1 day.
2. Add an inline policy to the instance role for that bucket:
   ```json
   { "Version": "2012-10-17", "Statement": [{
       "Effect": "Allow",
       "Action": ["s3:PutObject","s3:GetObject","s3:DeleteObject","s3:ListBucket"],
       "Resource": ["arn:aws:s3:::BUCKET","arn:aws:s3:::BUCKET/*"] }] }
   ```
3. Set `S3_BUCKET=<bucket>` in `.env` and redeploy. Leave it blank to disable.

## Option A — behind an existing reverse proxy (Caddy)
```bash
# DocMind joins the proxy's Docker network; publishes no host port.
PROXY_NETWORK=<proxy_network> DOCMIND_IMAGE=jash09/docmind:latest \
  docker compose -f deploy/docker-compose.behind-proxy.yml up -d
```
Find the network with `docker inspect <proxy-container> --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}'`.

Then add a site block to the proxy's Caddyfile and reload it:
```
your.domain {
        encode gzip
        reverse_proxy docmind:8000 {
                flush_interval -1
        }
}
```
```bash
docker exec <proxy-container> caddy reload --config /etc/caddy/Caddyfile
```
`flush_interval -1` keeps SSE token streaming working through the proxy.

## Option B — standalone (fresh box, DocMind owns 80/443)
```bash
docker compose -f deploy/docker-compose.prod.yml pull
docker compose -f deploy/docker-compose.prod.yml up -d
```
Set `DOMAIN=your.domain` in `.env`; Caddy auto-fetches the cert.

## DNS
Point your domain (DuckDNS, a real domain, or `<ip>.sslip.io`) at the box's
public IP — ideally an Elastic IP so it's stable. Ports 80/443 must be open.

## Verify
```bash
docker exec docmind python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/api/health').read())"
```
Then open `https://your.domain` — defaults to Bedrock, no key needed.

---

## CI/CD (optional, `.github/workflows/deploy.yml`)
Push-button (or merge-triggered) deploys: GitHub Actions builds the image,
pushes it to **Docker Hub**, then triggers the box over **SSM** to pull + restart.

Repo **secrets** required:
- `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` — to push the image.
- `AWS_DEPLOY_ROLE_ARN` — an IAM role trusting GitHub OIDC, allowed to
  `ssm:SendCommand`. (Set up the OIDC provider: IAM → Identity providers →
  `https://token.actions.githubusercontent.com`, audience `sts.amazonaws.com`.)
- `EC2_INSTANCE_ID` — the target instance.

`deploy/deploy.sh` runs on the box (pulls the image, `up -d`, prunes). Images are
tagged with the git SHA on Docker Hub, so any previous `jash09/docmind:<sha>`
can be re-deployed for rollback.
