# Deploying DocMind (CD to EC2 + ECR, HTTPS via DuckDNS)

This is the one-time setup behind the `Deploy (CD)` GitHub Actions workflow.
After it's done, deploying is a single click (or automatic on merge).

## The flow

```
push/merge ──▶ GitHub Actions
                 │  (1) OIDC: assume an AWS role — no stored keys
                 │  (2) docker build → push image to ECR
                 │  (3) SSM Run Command → tell EC2 to deploy
                 ▼
              EC2 box: git pull · ecr login · compose pull · up -d
                 ▼
              Caddy serves https://<you>.duckdns.org  (auto Let's Encrypt cert)
```

You provide three things via GitHub repo **Secrets**:
`AWS_DEPLOY_ROLE_ARN`, `EC2_INSTANCE_ID` (and the region/repo are in `deploy.yml`).

---

## 1. On the EC2 box (one time)

```bash
sudo mkdir -p /opt/docmind && sudo chown $USER /opt/docmind
git clone https://github.com/JashB-28/docmind.git /opt/docmind
cd /opt/docmind

# Docker, compose plugin, awscli, and the SSM agent (preinstalled on Amazon
# Linux 2023 / recent Ubuntu AMIs — install only if missing).
# Then create the production .env:
cp .env.example .env && nano .env
#   set real OPENAI_API_KEY / PINECONE_API_KEY (or rely on Bedrock + the role)
#   set DOMAIN=<you>.duckdns.org
```

Open **ports 80 and 443** in the instance's security group (inbound, 0.0.0.0/0).

## 2. ECR repository

```bash
aws ecr create-repository --repository-name docmind --region us-east-1
```

## 3. EC2 instance role (lets the box pull from ECR + be driven by SSM)

Attach an IAM role to the instance with these AWS-managed policies:
- `AmazonEC2ContainerRegistryReadOnly`  (pull images)
- `AmazonSSMManagedInstanceCore`         (receive deploy commands)
- `AmazonBedrockReadOnly` + Bedrock invoke (only if you use the Bedrock provider — see below)

## 4. GitHub → AWS keyless auth (OIDC)

1. **IAM → Identity providers → Add provider → OpenID Connect**
   - URL: `https://token.actions.githubusercontent.com`
   - Audience: `sts.amazonaws.com`
2. **Create a role** ("GitHubDeployRole") with this trust policy (locks it to your repo):
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Principal": { "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com" },
       "Action": "sts:AssumeRoleWithWebIdentity",
       "Condition": {
         "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
         "StringLike": { "token.actions.githubusercontent.com:sub": "repo:JashB-28/docmind:*" }
       }
     }]
   }
   ```
3. Give the role permission to **push to ECR** and **send SSM commands**:
   - `AmazonEC2ContainerRegistryPowerUser`
   - an inline policy allowing `ssm:SendCommand` on your instance + the
     `AWS-RunShellScript` document.
4. Copy the role ARN.

## 5. GitHub repo secrets

Repo → **Settings → Secrets and variables → Actions → New repository secret**:
- `AWS_DEPLOY_ROLE_ARN` = the role ARN from step 4
- `EC2_INSTANCE_ID` = e.g. `i-0abc123...`

## 6. DuckDNS (free domain)

1. Sign in at https://www.duckdns.org with GitHub/Google.
2. Create a subdomain, e.g. `docmind` → you get `docmind.duckdns.org`.
3. Set its IP to your EC2 **public IP** (the box's Elastic IP is best so it
   doesn't change on reboot).
4. Put `DOMAIN=docmind.duckdns.org` in the box's `.env`. Caddy does the rest
   (fetches + renews the HTTPS certificate automatically).

## 7. Deploy

- GitHub → **Actions → Deploy (CD) → Run workflow**. Watch it build, push, and
  roll out. Visit `https://docmind.duckdns.org`.
- To deploy **automatically on every merge to main**, uncomment the `push:`
  trigger at the top of `.github/workflows/deploy.yml`.

---

## Notes

- **First run on the box** can use the local build instead of ECR:
  `DOMAIN=... docker compose -f docker-compose.prod.yml up -d --build` (after a
  one-off `docker build -t docmind .`). The workflow takes over after that.
- **Rollback**: images are tagged with the git SHA in ECR, so you can re-deploy
  any previous `docmind:<sha>` by pulling that tag.
- **Logs**: `docker compose -f docker-compose.prod.yml logs -f docmind` (or ship
  the JSON logs to CloudWatch).
