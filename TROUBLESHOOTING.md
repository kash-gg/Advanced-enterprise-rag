# Troubleshooting Guide

## Common Setup Issues

### Python 3.13/3.14 Compatibility Error

**Error Message:**
```
spaCy depends on Pydantic v1
Pydantic v1 does NOT support Python ≥ 3.13 properly
```

**Cause:** You're running Python 3.13 or 3.14, which is too new for spaCy.

**Solution:**

1. **Install Python 3.12** (recommended):
   - Download from: https://www.python.org/downloads/release/python-3120/
   - During installation, check "Add Python to PATH"
   - You can have multiple Python versions installed

2. **Use Python 3.12 specifically**:
   ```powershell
   # Windows - specify Python 3.12 explicitly
   py -3.12 -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
   
   ```bash
   # Linux/Mac
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Verify your Python version**:
   ```bash
   python --version
   # Should show: Python 3.9.x, 3.10.x, 3.11.x, or 3.12.x
   ```

4. **Managing Multiple Python Versions**:
   
   **Option A: Keep Both (Recommended)**
   - No need to uninstall Python 3.13/3.14
   - Use Python launcher to specify version:
   ```powershell
   # Windows - use Python 3.12 specifically
   py -3.12 -m venv venv
   ```
   
   **Option B: Uninstall Python 3.13/3.14**
   - Windows: Settings → Apps → Installed apps → Search "Python 3.13" → Uninstall
   - Or: Control Panel → Programs → Uninstall a program
   - Then verify: `python --version` shows 3.12.x

### Why This Happens

- spaCy uses Pydantic v1 for data validation
- Pydantic v1 doesn't support Python 3.13+ due to internal changes in Python
- This is a known issue in the Python ecosystem
- The spaCy team is working on migrating to Pydantic v2

### Alternative: Skip spaCy (Not Recommended)

If you absolutely must use Python 3.13/3.14, you can temporarily remove spaCy:

1. Remove `spacy>=3.7.0` from `requirements.txt`
2. In `src/config.py`, set `GRAPH_ENABLE=false` in your `.env`
3. Note: This disables the knowledge graph feature

---

## Other Common Issues

### Virtual Environment Not Activating

**Windows:**
```powershell
# If you get execution policy error
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### FAISS Installation Fails

**Error:** `faiss-cpu` fails to install

**Solution:**
```bash
# Try installing build tools first
pip install --upgrade pip setuptools wheel
pip install faiss-cpu
```

### PyTorch Installation is Slow

**Cause:** PyTorch is a large package (~2GB)

**Solution:** Be patient, it can take 5-10 minutes depending on your internet speed.

### spaCy Model Download Fails

**Error:** `python -m spacy download en_core_web_sm` fails

**Solution:**
```bash
# Download manually
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

### Out of Memory During Installation

**Cause:** Not enough RAM

**Solution:**
```bash
# Install packages one at a time
pip install --no-cache-dir -r requirements.txt
```

---

## Getting Help

If you encounter other issues:

1. Check this troubleshooting guide
2. Verify your Python version is 3.9-3.12
3. Ensure you have 8GB+ RAM
4. Check GitHub Issues: https://github.com/kash-gg/Advanced-enterprise-rag/issues
5. Create a new issue with:
   - Python version (`python --version`)
   - Operating system
   - Full error message
   - Steps to reproduce

---

## Supported Configurations

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.9, 3.10, 3.11, 3.12 |
| OS | Windows 10+, Ubuntu 20.04+, macOS 11+ |
| RAM | 8GB minimum, 16GB recommended |
| Disk Space | 5GB free space |
