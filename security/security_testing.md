# Security Testing Guide — Windows PowerShell

> All commands use `Invoke-RestMethod` (built into PowerShell).
> It handles JSON without any quote-escaping issues.
>
> Before starting, open a terminal, navigate to the root of the cloned repository, and activate your virtual environment:
> ```powershell
> & .\.venv\Scripts\Activate.ps1
> ```

---

## Step 1 — Start the ML API (Terminal 1)

```powershell
cd ml_api
$env:JWT_SECRET_KEY = "dev-secret-change-in-production"
$env:PORT = "8002"
python ml_api.py
```

Wait for:
```
INFO:     ML API ready. Models loaded: ['hypertension', ...]
INFO:     Uvicorn running on http://0.0.0.0:8002
```

---

## Step 2 — Start the Intake API (Terminal 2)

```powershell
cd intake_api
$env:JWT_SECRET_KEY = "dev-secret-change-in-production"
$env:ML_API_URL = "http://localhost:8002"
$env:PORT = "8001"
python intake_api.py
```

Wait for:
```
INFO:     Intake API ready. ML_API_URL=http://localhost:8002
INFO:     Uvicorn running on http://0.0.0.0:8001
```

---

## Step 3 — Open a third terminal for testing

```powershell
& .\.venv\Scripts\Activate.ps1
```

All commands below go in this third terminal.

---

## Test 1 — Health check (public, no token needed)

```powershell
Invoke-RestMethod http://localhost:8001/health
Invoke-RestMethod http://localhost:8002/health
```

Expected intake: `status=ok, foods_loaded=7083`
Expected ml: `status=ok, models_loaded={hypertension...}`

---

## Test 2 — Login and get a JWT token

```powershell
$login = Invoke-RestMethod -Method POST -Uri http://localhost:8001/v1/login `
    -ContentType "application/json" `
    -Body '{"username":"admin","password":"admin-password"}'

$TOKEN = $login.token
Write-Host "Role : $($login.role)"
Write-Host "Token: $TOKEN"
```

Expected output:
```
Role : admin
Token: eyJhbGc...
```

`$TOKEN` is now set and ready to use in all commands below.

---

## Test 3 — Invalid login → 403

```powershell
try {
    Invoke-RestMethod -Method POST -Uri http://localhost:8001/v1/login `
        -ContentType "application/json" `
        -Body '{"username":"admin","password":"wrongpassword"}'
} catch {
    $_.ErrorDetails.Message
}
```

Expected: `{"error":"Forbidden","message":"Invalid credentials","status_code":403,...}`

---

## Test 4 — Access protected route WITHOUT a token → 401

```powershell
try {
    Invoke-RestMethod http://localhost:8001/users/test-id
} catch {
    $_.ErrorDetails.Message
}
```

Expected: `{"error":"Unauthorized","message":"Token is missing, invalid, or expired","status_code":401,...}`

---

## Test 5 — Register a user WITH a valid token → 201

```powershell
$headers = @{ Authorization = "Bearer $TOKEN" }

$newUser = Invoke-RestMethod -Method POST -Uri http://localhost:8001/users `
    -ContentType "application/json" `
    -Headers $headers `
    -Body '{"name":"Alice","age":35,"gender":2,"race_ethnicity":3,"education_level":4,"weight_kg":68.0,"height_cm":165.0}'

$USER_ID = $newUser.user_id
Write-Host "User ID: $USER_ID"
```

Expected: a `user_id` UUID is printed and stored in `$USER_ID`.

---

## Test 6 — Submit a food intake (end-to-end with ML prediction)

```powershell
$headers = @{ Authorization = "Bearer $TOKEN" }

Invoke-RestMethod -Method POST -Uri "http://localhost:8001/users/$USER_ID/intake" `
    -ContentType "application/json" `
    -Headers $headers `
    -Body '{"items":[{"food_name":"Chicken","grams":200},{"food_name":"Rice","grams":150}]}'
```

Expected: a record with `nutrient_totals` and a `prediction` block showing disease risk flags.

---

## Test 7 — RBAC failure — user token on a service-only route → 403

Get a `user`-role token:

```powershell
$userLogin = Invoke-RestMethod -Method POST -Uri http://localhost:8001/v1/login `
    -ContentType "application/json" `
    -Body '{"username":"user","password":"user-password"}'

$USER_TOKEN = $userLogin.token
```

Try to call ML API `/predict` directly with the user token:

```powershell
try {
    Invoke-RestMethod -Method POST -Uri http://localhost:8002/predict `
        -ContentType "application/json" `
        -Headers @{ Authorization = "Bearer $USER_TOKEN" } `
        -Body '{"age":40,"gender":1,"race_ethnicity":3,"education_level":3,"weight_kg":80,"height_cm":175,"protein_g":80,"carbs_g":200,"sugar_g":50,"fiber_g":20,"total_fat_g":60,"saturated_fat_g":20,"cholesterol_mg":250,"sodium_mg":2000}'
} catch {
    $_.ErrorDetails.Message
}
```

Expected: `{"error":"Forbidden","message":"Access requires role: service or admin","status_code":403,...}`

---

## Test 8 — Rate limit on login → 429 after 5 attempts

```powershell
1..7 | ForEach-Object {
    Write-Host -NoNewline "Attempt $_ -> "
    try {
        Invoke-RestMethod -Method POST -Uri http://localhost:8001/v1/login `
            -ContentType "application/json" `
            -Body '{"username":"admin","password":"admin-password"}' | Out-Null
        Write-Host "200 OK"
    } catch {
        $code = $_.Exception.Response.StatusCode.value__
        Write-Host "$code $($_.ErrorDetails.Message | ConvertFrom-Json | Select-Object -ExpandProperty error)"
    }
}
```

Expected:
```
Attempt 1 -> 200 OK
Attempt 2 -> 200 OK
Attempt 3 -> 200 OK
Attempt 4 -> 200 OK
Attempt 5 -> 200 OK
Attempt 6 -> 429 Too Many Requests
Attempt 7 -> 429 Too Many Requests
```

---

## Test 9 — Input validation → 400

Empty body:

```powershell
try {
    Invoke-RestMethod -Method POST -Uri http://localhost:8001/users `
        -ContentType "application/json" `
        -Headers @{ Authorization = "Bearer $TOKEN" } `
        -Body '{}'
} catch {
    $_.ErrorDetails.Message
}
```

Expected: `{"error":"Bad Request","message":"body.name: Field required; ...","status_code":400,...}`

Age out of range (must be 0–120):

```powershell
try {
    Invoke-RestMethod -Method POST -Uri http://localhost:8001/users `
        -ContentType "application/json" `
        -Headers @{ Authorization = "Bearer $TOKEN" } `
        -Body '{"name":"Bob","age":999,"gender":1,"race_ethnicity":1,"education_level":1,"weight_kg":70,"height_cm":175}'
} catch {
    $_.ErrorDetails.Message
}
```

Expected: `{"error":"Bad Request","message":"body.age: ...","status_code":400,...}`

---

## Test 10 — Unknown route → 404

```powershell
try {
    Invoke-RestMethod http://localhost:8001/not-a-route
} catch {
    $_.ErrorDetails.Message
}
```

Expected: `{"error":"Not Found",...}`

---

## Test 11 — Security headers check

```powershell
$resp = Invoke-WebRequest http://localhost:8001/health
$resp.Headers | Format-Table -AutoSize
```

Look for:
```
x-content-type-options : nosniff
x-frame-options        : DENY
x-xss-protection       : 1; mode=block
referrer-policy        : no-referrer
```

---

## Test 12 — View logs

Switch to Terminal 1 (ml-api) and Terminal 2 (intake-api) to see live logs:

```
INFO:     login_success username=admin role=admin ip=127.0.0.1
INFO:     method=POST path=/users status=201 latency_ms=2.3 ip=127.0.0.1 user=admin(admin)
WARNING:  auth_failure username=admin ip=127.0.0.1
WARNING:  rate_limit_exceeded ip=127.0.0.1 path=/v1/login
```

---

## OWASP ZAP scan (requires Docker Desktop)

With both services running, open a fourth terminal:

```powershell
docker pull ghcr.io/zaproxy/zaproxy:stable

docker run --rm -v "${PWD}:/zap/wrk/:rw" ghcr.io/zaproxy/zaproxy:stable zap-baseline.py -t http://host.docker.internal:8001 -r zap_report.html
```

Report saved to `zap_report.html` — open it in a browser.

---

## AKS end-to-end (after deployment)

```powershell
kubectl get ingress -n diet-risk
```

Replace `<EXTERNAL-IP>` and `<ADMIN_PASSWORD>`:

```powershell
$login = Invoke-RestMethod -Method POST -Uri http://<EXTERNAL-IP>/api/v1/login `
    -ContentType "application/json" `
    -Body '{"username":"admin","password":"<ADMIN_PASSWORD>"}'

$TOKEN = $login.token

Invoke-RestMethod -Method POST -Uri http://<EXTERNAL-IP>/api/users `
    -ContentType "application/json" `
    -Headers @{ Authorization = "Bearer $TOKEN" } `
    -Body '{"name":"Test","age":30,"gender":1,"race_ethnicity":1,"education_level":3,"weight_kg":75,"height_cm":178}'
```
