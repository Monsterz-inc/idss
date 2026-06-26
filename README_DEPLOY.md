Render deployment checklist

1. Create or use an existing Render Web Service connected to this repository.

2. Environment variables (Render dashboard → Service → Environment):
   - `EXTERNAL_URL`: Set this to the public service URL Render assigns (e.g., `https://your-service.onrender.com`).
   - `PPO_MODEL_URL`: (optional) Public URL to `ppo_idss.zip` (model artifact). If set, the app will download the model at startup and enable DRL features.
   - `PYTHONUNBUFFERED`: `1` (already set in `render.yaml`)

3. If you don't provide `PPO_MODEL_URL`, you can upload `ppo_idss.zip` as a release asset in GitHub and set `PPO_MODEL_URL` to that asset's URL, or host the zip on S3/Cloudflare/another static host.

4. Redeploy the service after setting environment variables.

5. Check the root endpoint `/` — it should return JSON with `ppo_ready: true` when the model is present and loaded, and `dashboard` should point to your `EXTERNAL_URL/dashboard`.

6. If the model fails to download, check build and runtime logs in the Render dashboard for messages like "Downloading PPO model" or "PPO model not available".

Quick steps to add `PPO_MODEL_URL` in the Render UI:

- Go to Render dashboard → Select service → Environment → Add Environment Variable
- Enter `PPO_MODEL_URL` and paste the public URL to `ppo_idss.zip`
- Save and click "Manual Deploy" or push to `main` to trigger an auto-deploy.

Alternatives:
- Upload `ppo_idss.zip` to GitHub Releases and use that URL.
- Store the model in an S3 bucket and grant public read access to the object.

Security note: Avoid committing large binaries to the repo; prefer releases or cloud storage.
