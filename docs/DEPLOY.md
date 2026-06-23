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

## CI/CD (`.github/workflows/deploy.yml`)
Push-button deploys: GitHub Actions builds the image, pushes it to **Docker Hub**,
then triggers the box over **SSM** to pull + restart. The workflow is
`workflow_dispatch` only (a manual **Run workflow** button) — deliberately, since
each live run spends real Bedrock money. To deploy automatically on every merge to
`main`, uncomment the `push:` trigger at the top of the file.

### Repo **secrets** required (Settings → Secrets and variables → Actions)
- `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` — to push the image. Create the token at
  Docker Hub → Account settings → Personal access tokens (Read & Write scope).
- `AWS_DEPLOY_ROLE_ARN` — an IAM role trusting GitHub OIDC, allowed `ssm:SendCommand`
  (see setup below).
- `EC2_INSTANCE_ID` — the target instance (e.g. `i-0abc…`).

### Optional repo **variable**
- `DEPLOY_DIR` — absolute path of the clone on the box. Defaults to
  `/home/ec2-user/docmind` if unset. Set it under Settings → Variables if your
  clone lives elsewhere.

### One-time AWS OIDC role setup
GitHub Actions assumes a short-lived role via OIDC — no long-lived keys stored.

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
INSTANCE_ID=i-0abc...                       # your box
REPO=JashB-28/docmind                       # owner/repo

# 1. Register GitHub as an OIDC identity provider (skip if it already exists).
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 || true

# 2. Trust policy: only this repo can assume the role.
cat > trust.json <<JSON
{ "Version": "2012-10-17", "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
      "StringLike":  { "token.actions.githubusercontent.com:sub": "repo:${REPO}:*" } } }] }
JSON
aws iam create-role --role-name docmind-deploy --assume-role-policy-document file://trust.json

# 3. Permission: send the deploy command to just this instance.
cat > perms.json <<JSON
{ "Version": "2012-10-17", "Statement": [
  { "Effect": "Allow", "Action": "ssm:SendCommand",
    "Resource": [
      "arn:aws:ec2:${REGION}:${ACCOUNT_ID}:instance/${INSTANCE_ID}",
      "arn:aws:ssm:${REGION}::document/AWS-RunShellScript" ] }] }
JSON
aws iam put-role-policy --role-name docmind-deploy \
  --policy-name ssm-send --policy-document file://perms.json

aws iam get-role --role-name docmind-deploy --query Role.Arn --output text   # → AWS_DEPLOY_ROLE_ARN
```

The instance also needs `AmazonSSMManagedInstanceCore` on its instance role and a
running SSM agent (Amazon Linux ships it) for the command to land.

### Box-side config (one-time, in the repo-root `.env` on the EC2 box)
`deploy/deploy.sh` defaults to the **behind-proxy** layout. Docker Compose reads
these from `.env` for the live deploy:
```
COMPOSE_FILE=deploy/docker-compose.behind-proxy.yml   # default; omit unless overriding
PROXY_NETWORK=scoutone_default
DOCMIND_IMAGE=jash09/docmind:latest
```

`deploy/deploy.sh` runs on the box (self-locates the repo, pulls the image,
`up -d`, prunes). Images are tagged with the git SHA on Docker Hub, so any
previous `jash09/docmind:<sha>` can be re-deployed for rollback.
