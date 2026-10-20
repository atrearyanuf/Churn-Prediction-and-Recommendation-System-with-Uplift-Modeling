# Batch pipeline: docker run churn-uplift {train,score,monitor} ...
FROM python:3.12-slim

# XGBoost and LightGBM need the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY functions/ functions/
COPY churn_uplift/ churn_uplift/
COPY config.toml streamlit_dashboard.py ./

RUN useradd --create-home --uid 1000 app && mkdir -p data artifacts outputs && chown -R app /app
USER app

ENTRYPOINT ["python", "-m", "churn_uplift"]
CMD ["--help"]
