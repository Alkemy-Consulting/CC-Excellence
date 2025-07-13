# GitHub Copilot Instructions - CC-Excellence Forecasting Tool

## Project Overview
This is a modular Streamlit-based contact center forecasting application with enterprise-grade time series analysis capabilities. The architecture follows a multi-page Streamlit app pattern with centralized state management and advanced Prophet/ARIMA/SARIMA forecasting models.

## Core Architecture

### Module Structure
- **`modules/`**: Core business logic and forecasting engines
  - `forecast_engine.py`: Central orchestrator for all forecasting models
  - `prophet_*.py`: Layered Prophet implementation (core, performance, diagnostics, presentation)
  - `*_enhanced.py`: Advanced implementations of ARIMA, SARIMA, Holt-Winters
  - `ui_components.py`: Reusable Streamlit UI components with configuration widgets
  - `data_utils.py`: Data preprocessing, validation, and sample data generation
  - `config.py`: Central configuration constants and mappings

### Session State Management
Critical state variables stored in `st.session_state`:
```python
# Data pipeline state
'data_loaded', 'cleaned_data', 'date_col', 'target_col'
# Model configuration state  
'model_configs', 'forecast_config', 'selected_model'
# Results state
'forecast_results_available', 'last_forecast_metrics', 'last_forecast_df'
```

### Multi-Page Navigation
- `app.py`: Main entry point with basic navigation
- `pages/1_📈Forecasting.py`: Primary forecasting interface with 3-tab layout
- `pages/2_🧮Capacity Sizing.py`, etc.: Additional contact center tools

## Key Development Patterns

### Prophet Architecture (Enterprise Pattern)
The Prophet implementation uses a sophisticated multi-layer architecture:
```python
# Core layer: ProphetForecaster, ProphetForecastResult dataclasses
from .prophet_core import ProphetForecaster
# Performance layer: Caching, optimization, monitoring
from .prophet_performance import OptimizedProphetForecaster, performance_monitor
# Diagnostics layer: Advanced model analysis
from .prophet_diagnostics import ProphetDiagnosticAnalyzer
```

### UI Component Pattern
UI components in `ui_components.py` follow this pattern:
```python
def render_*_config() -> Dict[str, Any]:
    """Returns configuration dict for model parameters"""
    # Always return Dict[str, Any] for model configs
    # Use st.expander for grouping related parameters
    # Include help text and validation
```

### State Management Workflow
1. **Data Loading**: `render_data_upload_section()` → updates `st.session_state.data_loaded`
2. **Configuration**: Various `render_*_config()` → updates `st.session_state.model_configs`
3. **Execution**: `run_enhanced_forecast()` → updates `st.session_state.forecast_results_available`
4. **Display**: Multi-tab results display with Advanced Diagnostic tab

## Critical Implementation Details

### Forecasting Engine Flow
```python
# Always use this pattern for model execution:
df, metrics, plots = run_enhanced_forecast(
    df, date_col, target_col, model_name, model_config, base_config
)
# Store results in session state for Advanced Diagnostic tab
st.session_state.last_forecast_metrics = metrics
st.session_state.last_forecast_df = df
```

### Error Handling Pattern
- Use `try/except` blocks with informative `st.error()` messages
- Validate data requirements before model execution
- Check minimum data points: `len(df) < 10` for basic, `< seasonal_periods * 2` for seasonal models

### Performance Optimization
- Prophet uses `@lru_cache` decorators for expensive operations
- `DataFrameOptimizer` for memory-efficient data handling
- Performance monitoring with `PerformanceMonitor` context manager

### Dependencies & Environment
- **Key packages**: `streamlit`, `prophet==1.1.7`, `pmdarima>=2.0.0`, `plotly`, `psutil`
- **Virtual environment**: Uses `.venv/` with specific package versions
- **Testing**: Extensive test files for each component (prefix `test_*`)

## Advanced Diagnostic Tab
The Advanced Diagnostic tab (`tab3` in Forecasting.py) provides comprehensive model analysis:
- Performance metrics interpretation (MAPE, R², MAE, RMSE)
- Residual analysis with Prophet diagnostics
- Forecast quality assessment with confidence intervals
- Time series decomposition and seasonality analysis
- Statistical logging and export functionality

## Development Guidelines & Standards

### Code Quality & Architecture
- **Clean old code**: Always review and remove legacy code or replaced functions
- **Stay on scope**: Never make changes outside the current request
- **Respect core principles**: Never modify against project's fundamental architecture
- **Enterprise readiness**: Code must be production-ready with scientific rigor in algorithms
- **Modularity**: Develop modular, scalable, and easily replicable functionality

### UX/UI Consistency Rules
- **Visual consistency**: Maintain same graphics for buttons across different model components
- **Single column layout**: All development must use single-column layout
- **Sidebar structure**: All parameters in vertical layout (no horizontal alignment)
- **No sidebar metrics**: Sidebar for parameters only, no metrics or charts
- **Parameter help**: Every input must have help text with business-friendly examples

### Module Development Workflow
```python
# Required pattern for new modules:
# 1. Define architectural details with user first
# 2. Specify data processing phases and file structure  
# 3. Design UX/UI layout before coding
# 4. All parameters in sidebar → launch button → main results display
```

### Critical Analysis Pattern
- **Deep prompt analysis**: Always analyze requirements against full system architecture
- **Technical specification**: Elaborate detailed technical requirements with correct function calls
- **Critical evaluation**: If inconsistencies found, recommend alternatives instead of proceeding
- **Architecture integration**: Ensure new code integrates with existing patterns

### Formatting Standards
- **Metrics precision**: All metrics display with exactly 2 decimal places
- **No new files**: Never create files without explicit user request
- **Function consistency**: Use established patterns from `ui_components.py`

## Common Anti-Patterns to Avoid
- Don't modify session state directly in UI components (return configs instead)
- Don't use global variables (use `modules/config.py` constants)
- Don't import modules circularly (follow the dependency hierarchy)
- Avoid hardcoded values (use `config.py` mappings like `DATE_FORMATS`, `FREQUENCY_MAP`)
- Never place multiple parameters horizontally in sidebar
- Never add charts or metrics to sidebar sections

## Testing & Debugging
- Run `python test_*.py` files for component testing
- Use `streamlit run app.py` for full application testing
- Debug state issues by checking `st.session_state` in sidebar
- Performance issues: Check `prophet_performance.py` monitoring output

## External Integrations
- Holiday data via `modules/holidays.py` with country-specific calendars
- Plotly for all visualizations (consistent styling via `PLOT_CONFIG`)
- CSV/Excel file upload with automatic column detection
- Export functionality for forecasts and diagnostic reports
