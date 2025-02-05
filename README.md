The dataset contains multiple sheets, each representing different time periods:

"4th period" (2005-2010)

"3rd period" (2000-2005)

"2nd period" (1995-2000)

"1st period" (1990-1995)

"all period" (1990-2010) – Covers the entire 20-year span

"Time frame" – Contains the timeline details



**Column Name Explanations**
ID – Unique identifier for each investment portfolio or customer.

Large B/P (Book-to-Price Ratio) – The ratio of a company's book value (net assets) to its market price.

Higher B/P → Possibly undervalued (value stocks).
Lower B/P → Growth stocks (higher expectations for future earnings).
Large ROE (Return on Equity) – Measures a company’s profitability relative to shareholders' equity.

Higher ROE → More profitable.
Used for evaluating investment performance.
Large S/P (Sales-to-Price Ratio) – Compares a company's total revenue (sales) to its market price.

Higher S/P → Indicates undervaluation.
Large Return Rate in the Last Quarter – Percentage return an investment achieved in the most recent quarter.

Higher return → Good short-term performance.
Large Market Value – The total market capitalization of the company or portfolio.

Larger values → Big companies (large-cap stocks).
Small Systematic Risk – Measures the portion of investment risk that is unavoidable (market-wide risk).

Smaller values → Less exposure to overall market volatility.
Annual Return – Percentage return generated by the investment in a year.

Higher annual return → Strong portfolio performance.
Excess Return – The return beyond a benchmark (e.g., S&P 500).

Positive excess return → Outperforms the market.
Systematic Risk – The risk inherent in the entire market (e.g., interest rates, inflation).

Higher systematic risk → More market-sensitive investment.
Total Risk – Combination of systematic risk (market-wide) and unsystematic risk (company-specific).
Higher total risk → More volatile investment.
Abs. Win Rate (Absolute Win Rate) – The percentage of periods where the investment had a positive return.
Higher win rate → More consistently profitable.
Rel. Win Rate (Relative Win Rate) – The win rate relative to a benchmark (e.g., S&P 500 or industry average).
Higher relative win rate → Outperforms the benchmark more often.

**Plan:**

Based on the explanations and dataset, here’s how these columns can fit into analysis for customer investment portfolios:

**Mapping to Analysis Needs**

ROI (Return on Investment):

Use “Annual Return” and “Excess Return” to calculate overall ROI.

AUM (Assets Under Management):

“Large Market Value” can serve as a proxy for AUM, representing the portfolio size.

Trend Analysis:

“Large Return Rate in the Last Quarter” and “Annual Return” can help analyze performance trends over time.

Investment Status:

Can be derived from performance metrics like “Annual Return” and “Abs. Win Rate” to categorize as Active/Underperforming.

Risk Level:

Use “Systematic Risk”, “Total Risk”, and “Small Systematic Risk” to assess risk levels (Low/Medium/High).

Categorization (Segmentation):

“Large B/P”, “Large ROE”, and “Large S/P” can help categorize stocks or portfolios (e.g., Growth vs. Value).

Performance Comparison:

“Rel. Win Rate” compared to a benchmark to evaluate if the portfolio consistently beats the market.

**Mapping Columns to Analysis Goals**

Customer & Portfolio Identifiers:

ID: Used for tracking individual portfolios or customers.

Financial Metrics (ROI, AUM):

Large B/P, Large ROE, Large S/P: Indicators of stock valuation and profitability; help assess portfolio quality.

Annual Return: Directly measures ROI.

Large Market Value: Can represent AUM if interpreted as portfolio size.

Excess Return: Shows performance relative to a benchmark, useful for ROI assessment.

Risk Indicators (Risk Level):

Small Systematic Risk: Indicates exposure to market-wide risks (Beta).

Systematic Risk & Total Risk: Essential for categorizing portfolios into Low/Medium/High risk.

Abs. Win Rate & Rel. Win Rate: Provide insights into consistency and benchmark performance.

Trend Analysis (Growth & Status):

Large Return Rate in the Last Quarter: Useful for quarterly trend analysis.

Annual Return: Tracks yearly performance trends.


**Plan of Action**

Preprocessing:

- Handle missing values.
- Normalize key financial columns.
- Categorize risk profiles based on risk metrics.
- Clustering (Customer Segmentation):

   Segment customers based on profitability, risk, and market value.
- Time Series Forecasting:

    Forecast portfolio performance using returns over time.
- Visualization:

  Build Tableau dashboards to visualize KPIs — ROI, risk levels, portfolio growth.
