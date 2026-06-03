# automated-dashboard-data-cleaning-from-excel
https://automated-dashboard-data-cleaning-from-excel-feszcnajkwbsywkhr.streamlit.app/
This project is a high-performance data engineering and visualization pipeline that transforms messy Excel data into a professional, interactive dashboard in real-time. It automates data cleaning using logic derived from advanced data science notebooks, ensuring that business insights are always based on clean, accurate data.
🌟 Key Features

   Real-Time Automation: A background Python engine monitors your Excel file. The moment you press "Save," the data is cleaned, imputed, and prepared for the dashboard.

  Smart Data Cleaning:

  Regex Processing: Automatically strips symbols ($, GB, Inches) and converts strings to numeric values.

  Advanced Imputation: Uses a hybrid hierarchy (Median and KNN Imputation) to fill missing values based on feature correlation.

  Outlier Detection: Automatically filters out "impossible" sales data using the Interquartile Range (IQR) method.

   Interactive Streamlit Dashboard:

  KPI Tracking: Instant visibility into Total Revenue, Units Sold, and Stock Levels.

  Price Analysis: Dynamic scatter plots comparing Price against hardware specs (RAM, CPU, etc.).

  Hierarchical Revenue: Brand and Model performance visualized through interactive Treemaps and Pie charts.

   Multi-Layer Filtering: Filter by Brand, OS, CPU, Color, and Special Features.

🛠️ The Technology Stack

  Language: Python 3.x

  Data Processing: Pandas, NumPy

  Machine Learning (Imputation): Scikit-Learn

  Dashboard: Streamlit

  Visualizations: Plotly Express

  Deployment: GitHub & Streamlit Community Cloud

🚀 How It Works

  Input: The user updates the data.xlsx file with new laptop inventory or sales figures.

   Process: The cleaner2.py script detects the file change. It handles data types, fixes typos, and predicts missing values.

   Output: The app.py dashboard (hosted on the web or locally) refreshes automatically to reflect the new data.
