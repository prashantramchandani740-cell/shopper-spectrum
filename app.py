

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="RetailIQ",
    page_icon="🛒",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.stApp{
    background-color:#0F172A;
    color:white;
}

.main-title{
    font-size:52px;
    font-weight:700;
    color:#38BDF8;
}

.sub-title{
    color:#CBD5E1;
    font-size:18px;
    margin-top:-15px;
}

.box{
    background:#1E293B;
    padding:25px;
    border-radius:12px;
    border-left:6px solid #38BDF8;
    margin-bottom:20px;
}

.small-box{
    background:#172554;
    padding:20px;
    border-radius:10px;
}

h2,h3{
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("online_retail.csv")

# ==========================================================
# DATA CLEANING
# ==========================================================

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Remove cancelled invoices
df = df[
    ~df["InvoiceNo"].astype(str).str.startswith("C")
]

# Remove negative quantities
df = df[
    df["Quantity"] > 0
]

# Remove zero or negative prices
df = df[
    df["UnitPrice"] > 0
]

# Remove POSTAGE
df = df[
    df["Description"] != "POSTAGE"
]

# Remove MANUAL
df = df[
    df["Description"] != "MANUAL"
]

# Remove blank descriptions
df = df[
    df["Description"].notna()
]

df = df[
    df["Description"].str.strip() != ""
]

# Remove duplicates
df = df.drop_duplicates()

# Reset Index
df = df.reset_index(drop=True)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

df["Revenue"] = (
    df["Quantity"] *
    df["UnitPrice"]
)

df["YearMonth"] = (
    df["InvoiceDate"]
    .dt.to_period("M")
    .astype(str)
)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🛒 RetailIQ</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Customer Analytics & Intelligent Product Recommendation Dashboard</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================================
# TABS
# ==========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Project Overview",
    "📊 Sales Intelligence",
    "👥 Customer Segmentation",
    "🎯 Product Recommendation"
])



# ==========================================================
# TAB 1
# ==========================================================

with tab1:

    left, right = st.columns([2,1])

    with left:

        st.markdown("""
<div class="box">

<h2>📌 Project Overview</h2>

Modern retail businesses generate millions of transactions every day.

Understanding customer purchasing behaviour helps businesses improve marketing, inventory planning, customer retention, and product recommendations.

RetailIQ converts raw ecommerce transaction data into meaningful business intelligence through data cleaning, exploratory data analysis, customer segmentation, and recommendation techniques.

</div>
""", unsafe_allow_html=True)

        st.markdown("## 🎯 Project Objectives")

        st.markdown("""

- Analyze customer purchasing behaviour

- Discover revenue trends

- Identify loyal customers

- Perform customer segmentation using RFM

- Recommend products using similarity analysis

- Support data-driven business decisions

""")

    with right:

        st.markdown("""
<div class="small-box">

<h3>💼 Business Applications</h3>

✅ Customer Segmentation

✅ Product Recommendation

✅ Marketing Campaigns

✅ Inventory Planning

✅ Cross-selling

✅ Upselling

✅ Revenue Analysis

</div>
""", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📂 Dataset Summary")

        dataset = pd.DataFrame({

            "Feature":[
                "Invoice Number",
                "Stock Code",
                "Description",
                "Quantity",
                "Invoice Date",
                "Unit Price",
                "Customer ID",
                "Country"
            ],

            "Description":[
                "Unique Invoice",
                "Product Code",
                "Product Name",
                "Items Purchased",
                "Transaction Date",
                "Price Per Unit",
                "Customer Identifier",
                "Customer Country"
            ]

        })

        st.table(dataset)

    with col2:

        st.subheader("⚙️ Project Workflow")

        st.markdown("""

1. Load Dataset

2. Data Cleaning

3. Exploratory Data Analysis

4. Feature Engineering

5. Customer Segmentation (RFM)

6. K-Means Clustering

7. Product Recommendation

8. Interactive Dashboard

""")

    st.divider()

    st.subheader("📈 Dataset Overview")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Transactions",
        f"{df['InvoiceNo'].nunique():,}"
    )

    metric2.metric(
        "Customers",
        f"{df['CustomerID'].nunique():,}"
    )

    metric3.metric(
        "Products",
        f"{df['StockCode'].nunique():,}"
    )

    metric4.metric(
        "Countries",
        f"{df['Country'].nunique():,}"
    )

    st.divider()

    

# ==========================================================
# TAB 2 : SALES INTELLIGENCE DASHBOARD
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------- LOAD DATA --------------------
with tab2:

    df.columns = df.columns.str.strip()

    # -------------------- CLEAN DATA --------------------

    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    df["Month"] = df["InvoiceDate"].dt.strftime("%b")
    df["Year"] = df["InvoiceDate"].dt.year
    df["Day"] = df["InvoiceDate"].dt.day_name()
    df["Date"] = df["InvoiceDate"].dt.date

    # -------------------- TITLE --------------------

    st.title("📊 Sales Intelligence Dashboard")
    st.caption("Discover hidden sales patterns and customer behaviour through interactive visual analytics.")

    st.markdown("---")

    st.subheader("🔍 Dashboard Filters")

    col1, col2 = st.columns(2)

    with col1:
        countries = sorted(df["Country"].unique())

    selected_country = st.multiselect(
    "Select Countries",
    countries,
    default=countries
    )


    with col2:

        start_date = df["InvoiceDate"].min().date()
    end_date = df["InvoiceDate"].max().date()

    selected_dates = st.date_input(
    "Select Date Range",
    [start_date, end_date]
    )

    # ---------------- APPLY FILTERS ----------------

    filtered_df = df[
    (df["Country"].isin(selected_country))
    ]

    if len(selected_dates) == 2:
        filtered_df = filtered_df[
    (filtered_df["InvoiceDate"].dt.date >= selected_dates[0]) &
    (filtered_df["InvoiceDate"].dt.date <= selected_dates[1])
    ]

    st.markdown("---")

    total_revenue = filtered_df["Revenue"].sum()

    total_orders = filtered_df["InvoiceNo"].nunique()

    customers = filtered_df["CustomerID"].nunique()

    products = filtered_df["StockCode"].nunique()

    avg_order = total_revenue / total_orders

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
    "💰 Revenue",
    f"£{total_revenue:,.0f}"
    )

    c2.metric(
    "🛒 Orders",
    f"{total_orders:,}"
    )

    c3.metric(
    "👥 Customers",
    f"{customers:,}"
    )

    c4.metric(
    "📦 Products",
    f"{products:,}"
    )

    c5.metric(
    "💳 Avg Order",
    f"£{avg_order:,.2f}"
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        daily_sales = (
    filtered_df
    .groupby("Date")["Revenue"]
    .sum()
    .reset_index()
    )

    fig = px.area(
    daily_sales,
    x="Date",
    y="Revenue",
    title="📈 Daily Revenue Trend"
    )

    fig.update_traces(
    line=dict(width=2)
    )

    fig.update_layout(
    height=420,
    xaxis_title="",
    yaxis_title="Revenue (£)"
    )

    st.plotly_chart(
    fig,
    use_container_width=True
    )

    with right:

        country_sales = (
    filtered_df
    .groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
    )

    fig = px.pie(
    country_sales,
    names="Country",
    values="Revenue",
    hole=0.55,
    title="🌍 Revenue Contribution by Country"
    )

    fig.update_layout(
    height=420
    )

    st.plotly_chart(
    fig,
    use_container_width=True
    )

    st.markdown("---")

    # ==========================================================
    # MONTHLY SALES HEATMAP
    # ==========================================================

    left, right = st.columns(2)

    with left:

        heatmap_data = (
    filtered_df
    .groupby(["Month", "Day"])["Revenue"]
    .sum()
    .reset_index()
    )

    month_order = [
    "Jan","Feb","Mar","Apr","May","Jun",
    "Jul","Aug","Sep","Oct","Nov","Dec"
    ]

    day_order = [
    "Monday","Tuesday","Wednesday",
    "Thursday","Friday","Saturday","Sunday"
    ]

    heatmap_data["Month"] = pd.Categorical(
    heatmap_data["Month"],
    categories=month_order,
    ordered=True
    )

    heatmap_data["Day"] = pd.Categorical(
    heatmap_data["Day"],
    categories=day_order,
    ordered=True
    )

    heatmap_data = heatmap_data.sort_values(["Month","Day"])

    fig = px.density_heatmap(
    heatmap_data,
    x="Month",
    y="Day",
    z="Revenue",
    color_continuous_scale="Blues",
    title="🔥 Sales Activity Heatmap"
    )

    fig.update_layout(height=450)

    st.plotly_chart(
    fig,
    use_container_width=True
    )

    # ==========================================================
    # PRODUCT TREEMAP
    # ==========================================================

    with right:

        top_products = (
    filtered_df
    .groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
    )

    fig = px.treemap(
    top_products,
    path=["Description"],
    values="Revenue",
    color="Revenue",
    color_continuous_scale="Viridis",
    title="📦 Top Revenue Generating Products"
    )

    fig.update_layout(height=450)

    st.plotly_chart(
    fig,
    use_container_width=True
    )


    # ==========================================================
    # CUSTOMER SPENDING ANALYSIS
    # ==========================================================

    left, right = st.columns(2)

    customer_summary = (
    filtered_df
    .groupby("CustomerID")
    .agg(
    TotalSpent=("Revenue","sum"),
    Orders=("InvoiceNo","nunique")
    )
    .reset_index()
    )

    # ==========================================================
    # CUSTOMER SPENDING ANALYSIS
    # ==========================================================

    left, right = st.columns(2)

    customer_summary = (
    filtered_df
    .groupby("CustomerID")
    .agg(
    TotalSpent=("Revenue", "sum"),
    Orders=("InvoiceNo", "nunique")
    )
    .reset_index()
    )

    with left:

       top15 = (
    customer_summary
    .sort_values("TotalSpent", ascending=True)   # Ascending for horizontal bars
    .tail(15)
    )

    fig = px.bar(
    top15,
    x="TotalSpent",
    y=top15["CustomerID"].astype(str),
    orientation="h",
    text="TotalSpent",
    title="🏆 Top 15 Customers by Spending",
    labels={
        "TotalSpent": "Revenue (£)",
        "CustomerID": "Customer ID"
        }
    )

    fig.update_traces(
    texttemplate="£%{x:,.0f}",
    textposition="outside",
    marker_color="#7B68EE"      # Single elegant purple
)

    fig.update_layout(
    height=650,
    showlegend=False,
    title_x=0.5,
    yaxis=dict(categoryorder="total ascending"),
    margin=dict(l=70, r=30, t=70, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)


    # ==========================================================
    # WEEKDAY SALES PERFORMANCE
    # ==========================================================

    with right:

        weekday_sales = (
    filtered_df
    .groupby("Day")["Revenue"]
    .sum()
    .reindex([
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
    ])
    .reset_index()
    )

    fig = px.bar(
    weekday_sales,
    x="Day",
    y="Revenue",
    color="Revenue",
    text_auto=".2s",
    title="📅 Revenue by Weekday"
    )

    fig.update_layout(
    height=430,
    xaxis_title="",
    yaxis_title="Revenue (£)"
    )

    st.plotly_chart(
    fig,
    use_container_width=True,
    key="weekday_sales_chart"
    )

    st.markdown("---")


    # ==========================================================
    # SALES TREND WITH MOVING AVERAGE
    # ==========================================================

    st.subheader("📈 Revenue Trend Analysis")

    trend = (
    filtered_df
    .groupby("Date")["Revenue"]
    .sum()
    .reset_index()
    )

    trend["Moving Average"] = trend["Revenue"].rolling(7).mean()

    fig = px.line(
    trend,
    x="Date",
    y=["Revenue", "Moving Average"],
    title="Revenue vs 7-Day Moving Average"
    )

    fig.update_layout(
    height=500,
    xaxis_title="Date",
    yaxis_title="Revenue (£)"
    )

    st.plotly_chart(
    fig,
    use_container_width=True,
    key="moving_average_chart"
    )

    st.markdown("---")


    # ==========================================================
    # ORDER VALUE DISTRIBUTION
    # ==========================================================

    left, right = st.columns(2)

    with left:

        invoice_values = (
    filtered_df
    .groupby("InvoiceNo")["Revenue"]
    .sum()
    .reset_index()
    )

    fig = px.histogram(
    invoice_values,
    x="Revenue",
    nbins=40,
    title="🛒 Distribution of Order Values"
    )

    fig.update_layout(
    height=430,
    xaxis_title="Order Value (£)"
    )

    st.plotly_chart(
    fig,
    use_container_width=True,
    key="order_distribution_chart"
    )


    # ==========================================================
    # TOP CUSTOMERS TABLE
    # ==========================================================

    with right:

        st.subheader("🏆 Highest Spending Customers")

    top_customers = (
    filtered_df
    .groupby("CustomerID")
    .agg(
    Total_Revenue=("Revenue", "sum"),
    Orders=("InvoiceNo", "nunique")
    )
    .sort_values("Total_Revenue", ascending=False)
    .head(10)
    )

    st.dataframe(
    top_customers.style.format({
    "Total_Revenue": "£{:,.2f}"
    }),
    use_container_width=True
    )

    st.markdown("---")

    # ==========================================================
    # BUSINESS INSIGHTS
    # ==========================================================

    st.subheader("📋 Dashboard Insights")

    highest_country = (
    filtered_df.groupby("Country")["Revenue"]
    .sum()
    .idxmax()
    )

    highest_product = (
    filtered_df.groupby("Description")["Revenue"]
    .sum()
    .idxmax()
    )

    best_customer = (
    filtered_df.groupby("CustomerID")["Revenue"]
    .sum()
    .idxmax()
    )

    col1, col2, col3 = st.columns(3)

    col1.success(
    f"""
    🌍 **Top Market**

    {highest_country}
    """
    )

    col2.info(
    f"""
    📦 **Best Selling Product**

    {highest_product}
    """
    )

    col3.warning(
    f"""
    👤 **Highest Spending Customer**

    {int(best_customer)}
    """
    )

    st.markdown("---")

    # ==========================================================
    # DASHBOARD FOOTER
    # ==========================================================

    st.caption(
    """
    This dashboard provides an overview of sales performance,
    customer purchasing behaviour, and product contribution
    using interactive visualizations built with Streamlit
    and Plotly.
    """
    )

# ===========================================================
# TAB 3 - CUSTOMER SEGMENTATION
# ===========================================================

with tab3:

    st.title("👥 Customer Segmentation")

    st.markdown("""
    ### Behavioral Customer Analysis

    Evaluate shoppers using **Recency, Frequency and Monetary (RFM)** metrics.
    The trained clustering model assigns each customer to a behavioural group,
    helping businesses identify loyal buyers, premium customers and customers
    who may require re-engagement.
    """)

    st.write("")

    # -------------------------------------------------------
    # SUB TABS
    # -------------------------------------------------------

    predict_tab, analysis_tab = st.tabs([
        "🧠 Customer Classifier",
        "📈 RFM Cluster Explorer"
    ])

    # =======================================================
    # CUSTOMER CLASSIFIER
    # =======================================================

    with predict_tab:

        st.subheader("Predict Customer Behaviour")

        st.write(
            "Enter the customer's purchasing information below to estimate "
            "the most appropriate customer category."
        )

        st.write("")

        col1, col2, col3 = st.columns(3)

        with col1:
            recency = st.number_input(
                "Days Since Last Order",
                min_value=0,
                value=30,
                step=1
            )

        with col2:
            frequency = st.number_input(
                "Total Purchase Count",
                min_value=1,
                value=5,
                step=1
            )

        with col3:
            monetary = st.number_input(
                "Total Customer Spend (£)",
                min_value=0.0,
                value=500.0,
                step=50.0
            )

        st.write("")

        predict = st.button(
            "🚀 Identify Customer Group",
            use_container_width=False
        )

        if predict:

            # --------------------------------------------------
            # Temporary prediction logic
            # Replace with your trained ML model later
            # --------------------------------------------------

            if monetary >= 2500 and frequency >= 15:
                segment = "Elite Customer"
                colour = "success"

            elif frequency >= 8:
                segment = "Loyal Shopper"
                colour = "info"

            elif recency >= 120:
                segment = "Dormant Customer"
                colour = "warning"

            else:
                segment = "Developing Customer"
                colour = "success"

            if colour == "success":
                st.success(f"Predicted Group : {segment}")

            elif colour == "info":
                st.info(f"Predicted Group : {segment}")

            else:
                st.warning(f"Predicted Group : {segment}")

            st.divider()

            left, right = st.columns(2)

            with left:
                st.metric(
                    "Days Since Last Purchase",
                    recency
                )

                st.metric(
                    "Purchase Frequency",
                    frequency
                )

            with right:
                st.metric(
                    "Customer Spending",
                    f"£{monetary:,.2f}"
                )

                st.markdown("### 💡 Recommended Business Action")

            if segment == "Elite Customer":

                st.success("""
                ✔ Offer premium membership benefits

                ✔ Provide early access to new arrivals

                ✔ Recommend high-value products

                ✔ Send exclusive reward vouchers
                """)

            elif segment == "Loyal Shopper":

                st.info("""
                ✔ Encourage repeat purchases

                ✔ Offer loyalty reward points

                ✔ Recommend complementary products

                ✔ Provide referral incentives
                """)

            elif segment == "Dormant Customer":

                st.warning("""
                ✔ Send win-back email campaigns

                ✔ Offer limited-time discounts

                ✔ Share personalised recommendations

                ✔ Encourage account reactivation
                """)

            else:

                st.success("""
                ✔ Introduce popular products

                ✔ Send welcome discount coupons

                ✔ Recommend trending categories

                ✔ Encourage another purchase
                """)

    # =======================================================
    # RFM ANALYSIS TAB
    # =======================================================

    with analysis_tab:

        st.subheader("📊 RFM Cluster Overview")

        st.write(
            "This section summarizes the customer clusters generated "
            "using the K-Means clustering algorithm."
        )

        st.write("")

        a, b, c, d = st.columns(4)

        with a:
            st.metric("Clusters", "4")

        with b:
            st.metric("Features", "R • F • M")

        with c:
            st.metric("Algorithm", "K-Means")

        with d:
            st.metric("Status", "Ready")

        st.divider()

        st.markdown("### Customer Groups")

        col1, col2 = st.columns(2)

        with col1:

            st.success("""
            ### 🏆 Elite Customers

            • Highest purchase value

            • Shop frequently

            • Most profitable customers

            • Priority retention group
            """)

            st.info("""
            ### 💎 Loyal Shoppers

            • Consistent purchasing behaviour

            • Good repeat purchase rate

            • Strong engagement

            • Excellent upselling opportunity
            """)

        with col2:

            st.warning("""
            ### 😴 Dormant Customers

            • Haven't purchased recently

            • Require re-engagement

            • At risk of churn

            • Best suited for promotional offers
            """)

            st.success("""
            ### 🌱 Developing Customers

            • New or occasional buyers

            • High future potential

            • Respond well to recommendations

            • Can become loyal customers
            """)

        st.divider()

        st.markdown("### 📈 Cluster Visualizations")

        left, right = st.columns(2)

        with left:
            st.info("Scatter Plot will appear here.")

        with right:
            st.info("Cluster Distribution Chart will appear here.")

        st.divider()

        st.caption(
            "Customer Intelligence Module • Built using Streamlit, "
            "Scikit-Learn and Pandas"
        )

        # ==========================================================
# TAB 4 : SMART PRODUCT SUGGESTION
# ==========================================================

with tab4:

    st.title("🛍️ Smart Product Suggestion")
    st.caption("Discover related products based on historical customer purchasing behaviour.")

    st.markdown("---")

    st.subheader("🔎 Product Recommendation Engine")

    st.write(
        """
        Select any product from the list below to explore items that are
        frequently purchased by similar customers. The recommendation engine
        uses customer buying patterns to identify closely related products.
        """
    )

    # -------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------

    retail = pd.read_csv("cleaned_online_retail.csv")

    retail = retail.dropna(subset=["Description"])

    retail["Description"] = retail["Description"].str.strip()

    retail = retail[retail["Description"] != ""]

    # -------------------------------------------------------
    # Product List
    # -------------------------------------------------------

    product_options = sorted(retail["Description"].unique())

    selected_product = st.selectbox(
        "Search or select a product",
        product_options,
        index=None,
        placeholder="Choose a product..."
    )

    st.write("")

        # -------------------------------------------------------
    # Build Customer-Product Matrix
    # -------------------------------------------------------

    purchase_matrix = retail.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )

    purchase_matrix = (purchase_matrix > 0).astype(int)

    similarity = cosine_similarity(purchase_matrix.T)

    similarity_df = pd.DataFrame(
        similarity,
        index=purchase_matrix.columns,
        columns=purchase_matrix.columns
    )

        # -------------------------------------------------------
    # Recommendation Button
    # -------------------------------------------------------

    if st.button("✨ Generate Suggestions"):

        if selected_product is None:

            st.warning("Please choose a product first.")

        else:

            recommended = (
                similarity_df[selected_product]
                .sort_values(ascending=False)
                .drop(selected_product)
                .head(5)
            )

            st.success(f"Products related to **{selected_product}**")

            recommendation_df = pd.DataFrame({
                "Suggested Product": recommended.index,
                "Similarity Score": np.round(recommended.values, 3)
            })

            st.dataframe(
                recommendation_df,
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---")

    st.info(
        "💡 Recommendations are generated using cosine similarity on customer purchase history."
    )

    

