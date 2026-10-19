#!/usr/bin/env node
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawn, spawnSync } = require("child_process");

const packageRoot = path.resolve(__dirname, "..");
const packageJson = JSON.parse(fs.readFileSync(path.join(packageRoot, "package.json"), "utf8"));

function userHome() {
  return process.env.HOME || process.env.USERPROFILE || os.homedir();
}

function expandHome(value) {
  if (!value) return value;
  if (value === "~") return userHome();
  if (value.startsWith("~/") || value.startsWith("~\\")) {
    return path.join(userHome(), value.slice(2));
  }
  return value;
}

function appHome() {
  return path.resolve(expandHome(process.env.AA_HOME || path.join(userHome(), ".agent-apprenticeship")));
}

function runCandidate(command, args) {
  return spawnSync(command, args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    env: process.env
  });
}

function pythonCheck(command) {
  const check = runCandidate(command, [
    "-c",
    "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
  ]);
  return {
    usable: !check.error && check.status === 0,
    error: check.error ? String(check.error.message || check.error) : null,
    status: check.status
  };
}

function findPython() {
  if (process.env.AA_PYTHON) {
    const candidate = expandHome(process.env.AA_PYTHON);
    const check = pythonCheck(candidate);
    if (check.usable) return candidate;
    console.error("AA_PYTHON is set but does not point to Python 3.11 or newer.");
    console.error(`AA_PYTHON=${candidate}`);
    console.error("");
    console.error("Agent Apprenticeship requires Python 3.11 or newer.");
    console.error("");
    console.error("macOS Homebrew:");
    console.error("brew install python@3.11");
    console.error("");
    console.error("Or set:");
    console.error("AA_PYTHON=/path/to/python3.11");
    process.exit(1);
  }
  const candidates = [
    "python3.11",
    "python3.12",
    "python3.13",
    "/opt/homebrew/bin/python3.11",
    "/opt/homebrew/bin/python3.12",
    "/opt/homebrew/bin/python3.13",
    "/usr/local/bin/python3.11",
    "/usr/local/bin/python3.12",
    "/usr/local/bin/python3.13",
    "python3",
    "python"
  ];
  for (const candidate of candidates) {
    if (pythonCheck(candidate).usable) return candidate;
  }
  return null;
}

function venvPython(venvDir) {
  return process.platform === "win32"
    ? path.join(venvDir, "Scripts", "python.exe")
    : path.join(venvDir, "bin", "python");
}

function runtimeEnv() {
  const env = { ...process.env };
  if (!env.PIP_CACHE_DIR) {
    env.PIP_CACHE_DIR = path.join(appHome(), "pip-cache");
  }
  return env;
}

function setupTimeout(defaultMs) {
  const raw = process.env.AA_NPM_SETUP_TIMEOUT_MS;
  if (!raw) return defaultMs;
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 ? value : defaultMs;
}

function appendLimited(current, chunk, limit = 16000) {
  const next = current + chunk.toString();
  return next.length > limit ? next.slice(next.length - limit) : next;
}

function setupHeartbeatMs(defaultMs) {
  const raw = process.env.AA_NPM_SETUP_HEARTBEAT_MS;
  if (!raw) return defaultMs;
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 ? value : defaultMs;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function acquireRuntimeLock(lockPath) {
  fs.mkdirSync(path.dirname(lockPath), { recursive: true });
  const started = Date.now();
  let lastHeartbeat = 0;
  const staleMs = setupTimeout(15 * 60 * 1000);
  const waitMs = setupTimeout(15 * 60 * 1000);
  while (true) {
    try {
      const fd = fs.openSync(lockPath, "wx");
      fs.writeFileSync(fd, JSON.stringify({
        pid: process.pid,
        createdAt: new Date().toISOString()
      }, null, 2) + "\n");
      return () => {
        try {
          fs.closeSync(fd);
        } catch (_) {
          // Best effort cleanup.
        }
        try {
          fs.unlinkSync(lockPath);
        } catch (_) {
          // Best effort cleanup.
        }
      };
    } catch (error) {
      if (!error || error.code !== "EEXIST") throw error;
      let stale = false;
      try {
        const stat = fs.statSync(lockPath);
        stale = Date.now() - stat.mtimeMs > staleMs;
      } catch (_) {
        stale = true;
      }
      if (stale) {
        try {
          fs.unlinkSync(lockPath);
        } catch (_) {
          // Another process may have refreshed or removed the lock.
        }
        continue;
      }
      if (Date.now() - started > waitMs) {
        console.error("Timed out waiting for another Agent Apprenticeship runtime setup to finish.");
        console.error(`Lock file: ${lockPath}`);
        process.exit(1);
      }
      if (Date.now() - lastHeartbeat > setupHeartbeatMs(30000)) {
        console.error("Waiting for another Agent Apprenticeship runtime setup to finish...");
        lastHeartbeat = Date.now();
      }
      await sleep(1000);
    }
  }
}

function runQuiet(command, args, label, timeoutMs) {
  return new Promise((resolve) => {
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let settled = false;
    const child = spawn(command, args, {
      stdio: ["ignore", "pipe", "pipe"],
      env: runtimeEnv()
    });
    const heartbeat = setInterval(() => {
      console.error(`Still working: ${label}...`);
    }, setupHeartbeatMs(30000));
    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
      setTimeout(() => {
        if (!settled) child.kill("SIGKILL");
      }, 3000).unref();
    }, setupTimeout(timeoutMs));
    child.stdout.on("data", (chunk) => {
      stdout = appendLimited(stdout, chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr = appendLimited(stderr, chunk);
    });
    child.on("error", (error) => {
      settled = true;
      clearInterval(heartbeat);
      clearTimeout(timeout);
      console.error(`Agent Apprenticeship runtime setup failed while trying to ${label}.`);
      console.error(String(error.message || error));
      process.exit(1);
    });
    child.on("close", (code, signal) => {
      settled = true;
      clearInterval(heartbeat);
      clearTimeout(timeout);
      if (code === 0) {
        resolve();
        return;
      }
      console.error(`Agent Apprenticeship runtime setup failed while trying to ${label}.`);
      if (timedOut || signal) {
        console.error("Runtime setup timed out. Re-run the command, or set AA_PYTHON to a working Python 3.11+ interpreter.");
      }
      const out = [stdout, stderr].filter(Boolean).join("\n").trim();
      if (out) console.error(out.slice(-8000));
      process.exit(code || 1);
    });
  });
}

function runVisible(command, args, label) {
  const result = spawnSync(command, args, {
    stdio: "inherit",
    env: runtimeEnv()
  });
  if (result.error || result.status !== 0) {
    console.error(`Agent Apprenticeship runtime setup failed while trying to ${label}.`);
    if (result.error) console.error(String(result.error.message || result.error));
    process.exit(result.status || 1);
  }
}

async function ensureRuntime(python) {
  const venvDir = path.resolve(process.env.AA_NPM_VENV || path.join(appHome(), "npm-venv", packageJson.version));
  const py = venvPython(venvDir);
  const markerPath = path.join(venvDir, ".agent-apprenticeship-npm.json");
  try {
    const marker = JSON.parse(fs.readFileSync(markerPath, "utf8"));
    if (marker.packageVersion === packageJson.version && fs.existsSync(py)) {
      return py;
    }
  } catch (_) {
    // Fall through and rebuild the runtime.
  }

  const releaseLock = await acquireRuntimeLock(`${venvDir}.install.lock`);
  try {
    try {
      const marker = JSON.parse(fs.readFileSync(markerPath, "utf8"));
      if (marker.packageVersion === packageJson.version && fs.existsSync(py)) {
        return py;
      }
    } catch (_) {
      // This process owns the setup lock and will rebuild the runtime.
    }

    console.error("Installing Agent Apprenticeship runtime...");
    console.error("First run can take a few minutes while Python dependencies are installed.");
    fs.rmSync(venvDir, { recursive: true, force: true });
    fs.mkdirSync(path.dirname(venvDir), { recursive: true });
    await runQuiet(python, ["-m", "venv", venvDir], "create the Python environment", 120000);
    await runQuiet(py, [
      "-m", "pip", "install",
      "--disable-pip-version-check",
      "--progress-bar", "off",
      "--no-input",
      "--retries", "10",
      "--timeout", "60",
      packageRoot
    ], "install the Python package", 600000);
    fs.writeFileSync(markerPath, JSON.stringify({
      packageName: packageJson.name,
      packageVersion: packageJson.version,
      installedAt: new Date().toISOString()
    }, null, 2) + "\n");
    console.error("Done.");
    return py;
  } finally {
    releaseLock();
  }
}

function runCli(python, args, usePackagedSource) {
  const env = { ...process.env };
  if (usePackagedSource) {
    const srcPath = path.join(packageRoot, "src");
    env.PYTHONPATH = env.PYTHONPATH ? `${srcPath}${path.delimiter}${env.PYTHONPATH}` : srcPath;
  }
  const result = spawnSync(python, ["-m", "agent_apprenticeship.cli", ...args], {
    stdio: "inherit",
    env
  });
  if (result.error) {
    console.error(`Failed to start Agent Apprenticeship: ${result.error.message || result.error}`);
    process.exit(1);
  }
  process.exit(result.status === null ? 1 : result.status);
}

(async () => {
  const python = findPython();
  if (!python) {
    console.error("Agent Apprenticeship requires Python 3.11 or newer.");
    console.error("");
    console.error("macOS Homebrew:");
    console.error("brew install python@3.11");
    console.error("");
    console.error("Or set:");
    console.error("AA_PYTHON=/path/to/python3.11");
    process.exit(1);
  }

  if (process.env.AA_NPM_USE_SYSTEM_PYTHON === "1") {
    runCli(python, process.argv.slice(2), true);
  }

  runCli(await ensureRuntime(python), process.argv.slice(2), false);
})().catch((error) => {
  console.error(`Failed to start Agent Apprenticeship: ${error.message || error}`);
  process.exit(1);
});
