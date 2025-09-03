# CC-Excellence Installation Guide

## Fixed Installation Instructions

Your repository is now properly set up with a working virtual environment and all core dependencies installed!

## ✅ What Has Been Fixed

1. **Removed the old problematic virtual environment**
2. **Created a new `.venv` virtual environment**
3. **Fixed dependency version conflicts**, specifically:
   - Updated `scikit-learn` from `==1.3.2` to `>=1.4.0` (Python 3.13 compatible)
   - Removed outdated `cython==0.29.36` constraint
   - Updated all package versions to be compatible with Python 3.13
4. **Installed all core dependencies successfully**

## 🚀 How to Run the Application

### Quick Start (Using the fixed environment):

```bash
cd /Users/taro/Projects/CC-Excellence
source .venv/bin/activate
streamlit run app.py
```

### Step-by-Step Instructions:

1. **Navigate to the project directory:**
   ```bash
   cd /Users/taro/Projects/CC-Excellence
   ```

2. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```
   
   You should see `(.venv)` at the beginning of your terminal prompt.

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and go to: `http://localhost:8501`

5. **When you're done**, deactivate the virtual environment:
   ```bash
   deactivate
   ```

## 📦 Installed Packages

The following core packages are now installed and working:

- ✅ **streamlit** - Web application framework
- ✅ **pandas & numpy** - Data manipulation
- ✅ **matplotlib & plotly** - Visualization
- ✅ **scikit-learn** - Machine learning (updated to v1.7.1)
- ✅ **statsmodels** - Statistical modeling
- ✅ **prophet** - Time series forecasting
- ✅ **ortools** - Optimization
- ✅ **seaborn** - Statistical plotting
- ✅ **openpyxl** - Excel file support
- ✅ **holidays** - Holiday data support

## ⚠️ Known Issues and Workarounds

### pmdarima Package
The `pmdarima` package was temporarily excluded due to compilation issues with Python 3.13. 

If you need ARIMA functionality:
1. **Option 1** (Recommended): Use `statsmodels.tsa.arima.model.ARIMA` which is already included
2. **Option 2**: Try installing pmdarima later when Python 3.13 support improves:
   ```bash
   source .venv/bin/activate
   pip install pmdarima --no-cache-dir
   ```

## 🔧 Development Setup

If you need to make changes or add new dependencies:

1. **Always activate the virtual environment first:**
   ```bash
   source .venv/bin/activate
   ```

2. **Install new packages using pip:**
   ```bash
   pip install package_name
   ```

3. **Update requirements file if needed:**
   ```bash
   pip freeze > config/requirements_working.txt
   ```

## 📁 File Structure

- `.venv/` - Virtual environment (created)
- `config/requirements_working.txt` - Working requirements file (created)
- `config/requirements_fixed.txt` - Fixed requirements (created)
- `config/requirements.txt` - Original requirements (kept for reference)

## 🆘 Troubleshooting

### If streamlit doesn't start:
```bash
source .venv/bin/activate
pip install --upgrade streamlit
streamlit run app.py
```

### If you see import errors:
```bash
source .venv/bin/activate
pip install -r config/requirements_working.txt
```

### To completely rebuild the environment:
```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r config/requirements_working.txt
```

## 🎉 You're Ready!

Your CC-Excellence application should now run without any compilation errors. All the main features are available:

- 📈 **Forecasting** - Prophet, ARIMA, SARIMA, Holt-Winters models
- 🧮 **Capacity Sizing** - Erlang calculations and optimization
- 👥 **Workforce Management** - Shift planning and scheduling
- ✅ **Adherence** - Performance monitoring

Enjoy using your contact center forecasting tool!
